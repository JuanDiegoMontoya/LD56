#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <tiny_obj_loader.h>

#include <Fwog/Context.h>
#include <Fwog/DebugMarker.h>
#include <Fwog/Shader.h>
#include <Fwog/Pipeline.h>
#include <Fwog/Buffer.h>

#include <miniaudio.h>

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>

#include <numeric>
#include <filesystem>
#include <optional>
#include <iostream>
#include <fstream>
#include <exception>
#include <thread>
#include <unordered_map>

using namespace JPH::literals;

// Globals
namespace
{
  namespace App
  {
    GLFWwindow* window{};
    glm::ivec2 windowSize{};
    bool cursorJustEnteredWindow = false;
    glm::dvec2 cursorPosPrev{};
    glm::dvec2 cursorOffset{};
  }

  namespace Render
  {
    using index_t = uint32_t;

    struct Vertex
    {
      glm::vec3 position{};
      glm::vec3 normal{};
      glm::vec2 texcoord{};
    };

    struct Mesh
    {
      std::optional<Fwog::TypedBuffer<Vertex>> vertexBuffer;
      std::optional<Fwog::TypedBuffer<index_t>> indexBuffer;
      uint32_t indexCount = 0;
    };

    struct InstanceUniforms
    {
      glm::mat4 world_from_object;
    };

    struct FrameUniforms
    {
      glm::mat4 clip_from_world;
    };

    std::optional<Fwog::GraphicsPipeline> pipeline;
    std::optional<Fwog::TypedBuffer<InstanceUniforms>> instanceBuffer;
    std::optional<Fwog::TypedBuffer<FrameUniforms>> frameUniformsBuffer;

    // Shaders
    namespace
    {
      const char* sceneVertexSource = R"(
#version 460 core

struct Vertex
{
  float px, py, pz;
  float nx, ny, nz;
  float tx, ty;
};

struct InstanceUniforms
{
  mat4 world_from_object;
};

layout(binding = 0, std430) readonly buffer VertexBuffer
{
  Vertex vertices[];
};

layout(binding = 1, std430) readonly buffer FrameUniforms
{
  mat4 clip_from_world;
}frame;

layout(binding = 2, std430) readonly buffer InstanceBuffer
{
  InstanceUniforms instances[];
};

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_texcoord;

void main()
{
  Vertex v = vertices[gl_VertexID];

  const vec3 posObj = vec3(v.px, v.py, v.pz);
  const vec3 normObj = vec3(v.nx, v.ny, v.nz);

  InstanceUniforms instance = instances[gl_InstanceID + gl_BaseInstance];

  v_position = (instance.world_from_object * vec4(posObj, 1.0)).xyz;
  v_normal   = transpose(inverse(mat3(instance.world_from_object))) * normObj;
  v_texcoord = vec2(v.tx, v.ty);

  gl_Position = frame.clip_from_world * vec4(v_position, 1.0);
}
)";

      const char* sceneFragmentSource = R"(
#version 460 core

layout(location = 0) out vec4 o_color;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_texcoord;

void main()
{
  o_color = vec4(v_normal * .5 + .5, 1.0);
  //o_color = vec4(1.0);
}
)";
    }
  }

  namespace Physics
  {
    namespace
    {
      namespace Layers
      {
        constexpr JPH::ObjectLayer NON_MOVING = 0;
        constexpr JPH::ObjectLayer MOVING = 1;
        constexpr JPH::ObjectLayer NUM_LAYERS = 2;
      };

      namespace BroadPhaseLayers
      {
        constexpr JPH::BroadPhaseLayer NON_MOVING(0);
        constexpr JPH::BroadPhaseLayer MOVING(1);
        constexpr JPH::uint NUM_LAYERS(2);
      };

      class ObjectLayerPairFilterImpl : public JPH::ObjectLayerPairFilter
      {
      public:
        bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override
        {
          switch (inObject1)
          {
          case Layers::NON_MOVING:
            return inObject2 == Layers::MOVING; // Non moving only collides with moving
          case Layers::MOVING:
            return true; // Moving collides with everything
          default:
            JPH_ASSERT(false);
            return false;
          }
        }
      };

      class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface
      {
      public:
        BPLayerInterfaceImpl()
        {
          // Create a mapping table from object to broad phase layer
          mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
          mObjectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
        }

        JPH::uint GetNumBroadPhaseLayers() const override
        {
          return BroadPhaseLayers::NUM_LAYERS;
        }

        JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override
        {
          JPH_ASSERT(inLayer < Layers::NUM_LAYERS);
          return mObjectToBroadPhase[inLayer];
        }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
        const char* GetBroadPhaseLayerName(BroadPhaseLayer inLayer) const override
        {
          switch ((BroadPhaseLayer::Type)inLayer)
          {
          case (BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING:	return "NON_MOVING";
          case (BroadPhaseLayer::Type)BroadPhaseLayers::MOVING:		return "MOVING";
          default: JPH_ASSERT(false); return "INVALID";
          }
        }
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

      private:
        JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::NUM_LAYERS];
      };

      class ObjectVsBroadPhaseLayerFilterImpl : public JPH::ObjectVsBroadPhaseLayerFilter
      {
      public:
        bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override
        {
          switch (inLayer1)
          {
          case Layers::NON_MOVING:
            return inLayer2 == BroadPhaseLayers::MOVING;
          case Layers::MOVING:
            return true;
          default:
            JPH_ASSERT(false);
            return false;
          }
        }
      };
    }

    struct HashBodyID
    {
      std::size_t operator()(const JPH::BodyID& body) const noexcept
      {
        return std::hash<JPH::uint32>{}(body.GetIndexAndSequenceNumber());
      }
    };

    JPH::TempAllocatorImpl* tempAllocator{}; 
    JPH::JobSystemThreadPool* jobSystem{};
    auto broad_phase_layer_interface = BPLayerInterfaceImpl();
    auto object_vs_broadphase_layer_filter = ObjectVsBroadPhaseLayerFilterImpl();
    auto object_vs_object_layer_filter = ObjectLayerPairFilterImpl();
    auto engine = JPH::PhysicsSystem();
    JPH::BodyInterface* body_interface{};
    std::unordered_map<JPH::BodyID, entt::entity, HashBodyID> bodyToEntity;
  }

  namespace Game
  {
    namespace ECS
    {
      // Components below here
      struct Name
      {
        std::string string;
      };

      struct Transform
      {
        glm::vec3 position = { 0, 0, 0 };
        glm::quat rotation = { 1, 0, 0, 0 };
        glm::vec3 scale = { 1, 1, 1 };
      };

      struct MeshRef
      {
        Render::Mesh* mesh{};
      };

      struct PhysicsKinematic
      {
        JPH::BodyID body;
      };

      struct PhysicsDynamic
      {
        JPH::BodyID body;
      };

      struct PhysicsKinematicAdded {};
      struct PhysicsDynamicAdded {};
    }

    struct View
    {
      glm::vec3 position{};
      float pitch{}; // pitch angle in radians
      float yaw = -glm::half_pi<float>();   // yaw angle in radians

      glm::vec3 GetForwardDir() const
      {
        return glm::vec3{ cos(pitch) * cos(yaw), sin(pitch), cos(pitch) * sin(yaw) };
      }

      glm::mat4 GetViewMatrix() const
      {
        // TODO: express view matrix without lookAt
        return glm::lookAt(position, position + GetForwardDir(), glm::vec3(0, 1, 0));
      }
    };


    View mainCamera{};
    float cursorSensitivity = 0.0025f;
    float cameraSpeed = 4.5f;

    // Game objects
    entt::registry registry;
    Render::Mesh testMesh;
    entt::entity testEntity;
  }
}

std::filesystem::path GetDataDirectory()
{
  static std::optional<std::filesystem::path> assetsPath;
  if (!assetsPath)
  {
    // Search for directory starting at the cwd and going up. Should work in both the debugger and shipped builds.
    auto dir = std::filesystem::current_path();
    while (!dir.empty())
    {
      auto maybeAssets = dir / "data";
      if (exists(maybeAssets) && is_directory(maybeAssets))
      {
        assetsPath = maybeAssets;
        break;
      }

      if (!dir.has_parent_path())
      {
        break;
      }

      dir = dir.parent_path();
    }
  }
  return assetsPath.value(); // Will throw if directory wasn't found.
}

std::string LoadFileText(const std::filesystem::path& path)
{
  std::ifstream file{ path };
  return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
}

std::pair<std::unique_ptr<std::byte[]>, std::size_t> LoadFileBinary(const std::filesystem::path& path)
{
  std::size_t fsize = std::filesystem::file_size(path);
  auto memory = std::make_unique<std::byte[]>(fsize);
  std::ifstream file{ path, std::ifstream::binary };
  std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), reinterpret_cast<char*>(memory.get()));
  return { std::move(memory), fsize };
}

// Callbacks
namespace
{
  void GLAPIENTRY OpenglErrorCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    [[maybe_unused]] GLsizei length,
    const GLchar* message,
    [[maybe_unused]] const void* userParam)
  {
    // Ignore certain verbose info messages (particularly ones on Nvidia).
    if (id == 131169 ||
      id == 131185 || // NV: Buffer will use video memory
      id == 131218 ||
      id == 131204 || // Texture cannot be used for texture mapping
      id == 131222 ||
      id == 131154 || // NV: pixel transfer is synchronized with 3D rendering
      id == 0         // gl{Push, Pop}DebugGroup
      )
      return;

    std::stringstream errStream;
    errStream << "OpenGL Debug message (" << id << "): " << message << '\n';

    switch (source)
    {
    case GL_DEBUG_SOURCE_API: errStream << "Source: API"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM: errStream << "Source: Window Manager"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER: errStream << "Source: Shader Compiler"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY: errStream << "Source: Third Party"; break;
    case GL_DEBUG_SOURCE_APPLICATION: errStream << "Source: Application"; break;
    case GL_DEBUG_SOURCE_OTHER: errStream << "Source: Other"; break;
    }

    errStream << '\n';

    switch (type)
    {
    case GL_DEBUG_TYPE_ERROR: errStream << "Type: Error"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: errStream << "Type: Deprecated Behaviour"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: errStream << "Type: Undefined Behaviour"; break;
    case GL_DEBUG_TYPE_PORTABILITY: errStream << "Type: Portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE: errStream << "Type: Performance"; break;
    case GL_DEBUG_TYPE_MARKER: errStream << "Type: Marker"; break;
    case GL_DEBUG_TYPE_PUSH_GROUP: errStream << "Type: Push Group"; break;
    case GL_DEBUG_TYPE_POP_GROUP: errStream << "Type: Pop Group"; break;
    case GL_DEBUG_TYPE_OTHER: errStream << "Type: Other"; break;
    }

    errStream << '\n';

    switch (severity)
    {
    case GL_DEBUG_SEVERITY_HIGH: errStream << "Severity: high"; break;
    case GL_DEBUG_SEVERITY_MEDIUM: errStream << "Severity: medium"; break;
    case GL_DEBUG_SEVERITY_LOW: errStream << "Severity: low"; break;
    case GL_DEBUG_SEVERITY_NOTIFICATION: errStream << "Severity: notification"; break;
    }

    std::cout << errStream.str() << '\n';
  }

  void CursorPosCallback(GLFWwindow*, double cursorX, double cursorY)
  {
    if (App::cursorJustEnteredWindow)
    {
      App::cursorPosPrev.x = cursorX;
      App::cursorPosPrev.y = cursorY;
      App::cursorJustEnteredWindow = false;
    }

    App::cursorOffset.x += cursorX - App::cursorPosPrev.x;
    App::cursorOffset.y += App::cursorPosPrev.y - cursorY;
    App::cursorPosPrev = { cursorX, cursorY };
  }

  void CursorEnterCallback(GLFWwindow*, int entered)
  {
    if (entered)
    {
      App::cursorJustEnteredWindow = true;
    }
  }

  void WindowResizeCallback(GLFWwindow* window, int, int)
  {
    int newWidth{};
    int newHeight{};
    glfwGetFramebufferSize(window, &newWidth, &newHeight);
    App::windowSize = { newWidth, newHeight };

    if (newWidth > 0 && newHeight > 0)
    {
      // TODO: invoke resize

      //app->OnWindowResize(app->windowWidth, app->windowHeight);
      //app->Draw(0);
    }
  }
}

// Application
namespace
{
  void InitializeApplication()
  {
    if (glfwInit() != GLFW_TRUE)
    {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwSetErrorCallback([](int, const char* msg) { printf("GLFW error: %s\n", msg); });

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_MAXIMIZED, false);
    glfwWindowHint(GLFW_DECORATED, true);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor)
    {
      throw std::runtime_error("Failed to find monitor");
    }
    const GLFWvidmode* videoMode = glfwGetVideoMode(monitor);
    // TODO: update app title
    App::window = glfwCreateWindow(static_cast<int>(videoMode->width * .75), static_cast<int>(videoMode->height * .75), "MyMfFrogEngine", nullptr, nullptr);
    if (!App::window)
    {
      glfwTerminate();
      throw std::runtime_error("Failed to create window");
    }

    glfwGetFramebufferSize(App::window, &App::windowSize.x, &App::windowSize.y);

    int monitorLeft{};
    int monitorTop{};
    glfwGetMonitorPos(monitor, &monitorLeft, &monitorTop);

    // Center window on the monitor
    glfwSetWindowPos(App::window, videoMode->width / 2 - App::windowSize.x / 2 + monitorLeft, videoMode->height / 2 - App::windowSize.y / 2 + monitorTop);

    glfwMakeContextCurrent(App::window);

    // vsync
    glfwSwapInterval(1);

    glfwSetCursorPosCallback(App::window, CursorPosCallback);
    glfwSetCursorEnterCallback(App::window, CursorEnterCallback);
    glfwSetFramebufferSizeCallback(App::window, WindowResizeCallback);

    //auto fwogCallback = [](std::string_view msg) { printf("Fwog: %.*s\n", static_cast<int>(msg.size()), msg.data()); };
    auto fwogCallback = nullptr;
    Fwog::Initialize({
      .glLoadFunc = glfwGetProcAddress,
      .verboseMessageCallback = fwogCallback,
      });

    // Set up the GL debug message callback.
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(OpenglErrorCallback, nullptr);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(App::window, true);
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();

    //glfwSetInputMode(App::window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetInputMode(App::window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  }

  void TerminateApplication()
  {
    glfwTerminate();
  }
}

// Graphics
namespace
{
  Render::Mesh LoadObjFile(const std::filesystem::path& path)
  {
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path.string()))
    {
      std::cout << "TinyObjReader error: " << reader.Error() << '\n';
      throw std::runtime_error("Failed to parse obj");
    }

    if (!reader.Warning().empty())
    {
      std::cout << "TinyObjReader warning: " << reader.Warning() << '\n';
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    //auto& materials = reader.GetMaterials();

    auto vertices = std::vector<Render::Vertex>();

    // Loop over shapes
    for (const auto& shape : shapes)
    {
      // Loop over faces(polygon)
      size_t index_offset = 0;
      for (const auto& fv : shape.mesh.num_face_vertices)
      {
        // Loop over vertices in the face.
        for (size_t v = 0; v < fv; v++)
        {
          auto vertex = Render::Vertex{};

          // access to vertex
          tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
          tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
          tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
          tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

          vertex.position = { vx, vy, vz };

          // Check if `normal_index` is zero or positive. negative = no normal data
          if (idx.normal_index >= 0)
          {
            tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
            tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
            tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
            vertex.normal = { nx, ny, nz };
          }

          // Check if `texcoord_index` is zero or positive. negative = no texcoord data
          if (idx.texcoord_index >= 0)
          {
            tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
            tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
            vertex.texcoord = { tx, ty };
          }

          // Optional: vertex colors
          // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
          // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
          // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];

          vertices.push_back(vertex);
        }
        index_offset += fv;
      }
    }

    auto indices = std::vector<Render::index_t>(vertices.size());
    std::iota(indices.begin(), indices.end(), 0);

    auto mesh = Render::Mesh{};
    mesh.indexCount = (uint32_t)indices.size();
    mesh.indexBuffer.emplace(indices);
    mesh.vertexBuffer.emplace(vertices);
    return mesh;
  }

  void InitializeRenderer()
  {
    auto vertexShader = Fwog::Shader(Fwog::PipelineStage::VERTEX_SHADER, Render::sceneVertexSource);
    auto fragmentShader = Fwog::Shader(Fwog::PipelineStage::FRAGMENT_SHADER, Render::sceneFragmentSource);
    Render::pipeline = Fwog::GraphicsPipeline({
      .vertexShader = &vertexShader,
      .fragmentShader = &fragmentShader,
      .rasterizationState = {.cullMode = Fwog::CullMode::NONE,},
      .depthState = {.depthTestEnable = true, .depthWriteEnable = true,},
      });

    Render::frameUniformsBuffer.emplace(Fwog::BufferStorageFlag::DYNAMIC_STORAGE);
  }

  void TerminateRenderer()
  {
    Render::instanceBuffer.reset();
    Render::frameUniformsBuffer.reset();
    Render::pipeline.reset();
    Fwog::Terminate();
  }

  void TickRender([[maybe_unused]] double dt)
  {
    auto renderMarker = Fwog::ScopedDebugMarker("TickRender");

    glEnable(GL_FRAMEBUFFER_SRGB);

    Fwog::RenderToSwapchain(
      Fwog::SwapchainRenderInfo{
        .name = "Render Triangle",
        .viewport = Fwog::Viewport{.drawRect{.offset = {0, 0}, .extent = {(uint32_t)App::windowSize.x, (uint32_t)App::windowSize.y}}},
        .colorLoadOp = Fwog::AttachmentLoadOp::CLEAR,
        .clearColorValue = {.2f, .0f, .2f, 1.0f},
        .depthLoadOp = Fwog::AttachmentLoadOp::CLEAR,
        .clearDepthValue = 1.0f,
      },
      [&]
      {
        auto projection = glm::perspective(glm::radians(60.0f), (float)App::windowSize.x / (float)App::windowSize.y, 0.1f, 100.0f);
        Render::frameUniformsBuffer->UpdateData(Render::FrameUniforms{.clip_from_world = projection * Game::mainCamera.GetViewMatrix()});

        struct Instance
        {
          size_t index;
          Render::Mesh* mesh;
        };

        auto instanceData = std::vector<Render::InstanceUniforms>();
        auto instances = std::vector<Instance>();

        auto view = Game::registry.view<Game::ECS::Transform, Game::ECS::MeshRef>();
        for (auto&& [entity, transform, meshRef] : view.each())
        {
          instances.emplace_back(instanceData.size(), meshRef.mesh);
          instanceData.emplace_back(glm::translate(glm::identity<glm::mat4>(), transform.position) * glm::mat4_cast(transform.rotation) * glm::scale(glm::identity<glm::mat4>(), transform.scale));
        }

        if (!Render::instanceBuffer || Render::instanceBuffer->Size() < instances.size() * sizeof(Render::InstanceUniforms))
        {
          Render::instanceBuffer.emplace(instanceData.size() * 3 / 2, Fwog::BufferStorageFlag::DYNAMIC_STORAGE, "Instance Data");
        }

        if (!instanceData.empty())
        {
          Render::instanceBuffer->UpdateData(instanceData);
        }

        Fwog::Cmd::BindGraphicsPipeline(Render::pipeline.value());
        Fwog::Cmd::BindStorageBuffer(1, Render::frameUniformsBuffer.value());
        Fwog::Cmd::BindStorageBuffer(2, Render::instanceBuffer.value());

        for (size_t i = 0; i < instances.size(); i++)
        {
          const auto& instance = instances[i];
          Fwog::Cmd::BindIndexBuffer(instance.mesh->indexBuffer.value(), Fwog::IndexType::UNSIGNED_INT);
          Fwog::Cmd::BindStorageBuffer(0, instance.mesh->vertexBuffer.value());
          Fwog::Cmd::DrawIndexed(instance.mesh->indexCount, 1, 0, 0, (uint32_t)i);
        }
      });

    ImGui::Render();
    auto* imguiDrawData = ImGui::GetDrawData();
    if (imguiDrawData->CmdListsCount > 0)
    {
      auto marker = Fwog::ScopedDebugMarker("UI");
      glDisable(GL_FRAMEBUFFER_SRGB);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      ImGui_ImplOpenGL3_RenderDrawData(imguiDrawData);
    }
    glfwSwapBuffers(App::window);
  }

  void TickUI(double dt)
  {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Text("FPS: %.0f", 1 / dt);
    ImGui::Text("Camera pos: (%.1f, %.1f, %.1f)", Game::mainCamera.position.x, Game::mainCamera.position.y, Game::mainCamera.position.z);
    ImGui::Text("Camera dir: (%.1f, %.1f, %.1f)", Game::mainCamera.GetForwardDir().x, Game::mainCamera.GetForwardDir().y, Game::mainCamera.GetForwardDir().z);
    ImGui::Text("Camera euler: yaw: %.2f, pitch: %.2f", Game::mainCamera.yaw, Game::mainCamera.pitch);
  }
}

// Physics
namespace
{
  void AddBodyKinematic(entt::entity e)
  {
    Game::registry.emplace<Game::ECS::PhysicsKinematicAdded>(e);
  }

  void AddBodyDynamic(entt::entity e)
  {
    Game::registry.emplace<Game::ECS::PhysicsDynamicAdded>(e);
  }

  void RemoveBodyKinematic(entt::entity e)
  {
    const auto& body = Game::registry.get<Game::ECS::PhysicsKinematic>(e).body;
    Physics::body_interface->RemoveBody(body);
    Physics::bodyToEntity.erase(body);
  }

  void RemoveBodyDynamic(entt::entity e)
  {
    const auto& body = Game::registry.get<Game::ECS::PhysicsDynamic>(e).body;
    Physics::body_interface->RemoveBody(body);
    Physics::bodyToEntity.erase(body);
  }

  void InitializePhysics()
  {
    Game::registry.on_construct<Game::ECS::PhysicsKinematic>().connect<AddBodyKinematic>();
    Game::registry.on_construct<Game::ECS::PhysicsDynamic>().connect<AddBodyDynamic>();
    Game::registry.on_destroy<Game::ECS::PhysicsKinematic>().connect<RemoveBodyKinematic>();
    Game::registry.on_destroy<Game::ECS::PhysicsDynamic>().connect<RemoveBodyDynamic>();

    JPH::RegisterDefaultAllocator();
    //JPH::Trace =
    //JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = )
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    Physics::tempAllocator = new JPH::TempAllocatorImpl(10 * 1024 * 1024);
    Physics::jobSystem = new JPH::JobSystemThreadPool(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, std::thread::hardware_concurrency() - 1);
    
    constexpr JPH::uint cMaxBodies = 1024;
    constexpr JPH::uint cNumBodyMutexes = 0;
    constexpr JPH::uint cMaxBodyPairs = 1024;
    constexpr JPH::uint cMaxContactConstraints = 1024;
    Physics::engine.Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints, Physics::broad_phase_layer_interface, Physics::object_vs_broadphase_layer_filter, Physics::object_vs_object_layer_filter);

    Physics::body_interface = &Physics::engine.GetBodyInterface();
    
    // Add static floor
    auto floor_shape_settings = JPH::BoxShapeSettings(JPH::Vec3(100, 1, 100));
    floor_shape_settings.SetEmbedded();
    auto floor_shape_result = floor_shape_settings.Create();
    auto floor_shape = floor_shape_result.Get();
    auto floor_settings = JPH::BodyCreationSettings(floor_shape, JPH::RVec3(0.0_r, -1.0_r, 0.0_r), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Physics::Layers::NON_MOVING);
    auto* floor = Physics::body_interface->CreateBody(floor_settings); // Note that if we run out of bodies this can return nullptr
    floor->SetRestitution(.5f);
    Physics::body_interface->AddBody(floor->GetID(), JPH::EActivation::DontActivate);

    Physics::engine.OptimizeBroadPhase();
  }

  void TerminatePhysics()
  {
    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
  }
}

// Game
namespace
{
  void InitializeGame()
  {
    Game::testEntity = Game::registry.create();
    Game::registry.emplace<Game::ECS::Name>(Game::testEntity).string = "Hello";
    auto& transform = Game::registry.emplace<Game::ECS::Transform>(Game::testEntity);

    transform.position = { 0, 0, -1 };

    Game::testMesh = LoadObjFile(GetDataDirectory() / "models/bunny.obj");

    //Render::Vertex vertices[] = {
    //  Render::Vertex{{0, 0, 0}},
    //  Render::Vertex{{1, -1, 0}},
    //  Render::Vertex{{1, 1, 0}},
    //};
    //uint32_t indices[] = { 0, 1, 2 };

    //Game::testMesh.vertexBuffer.emplace(std::span<const Render::Vertex>(vertices));
    //Game::testMesh.indexBuffer.emplace(std::span<const uint32_t>(indices));
    //Game::testMesh.indexCount = std::size(indices);

    Game::registry.emplace<Game::ECS::MeshRef>(Game::testEntity).mesh = &Game::testMesh;

    // Add moving sphere
    auto sphere_settings = JPH::BodyCreationSettings(new JPH::SphereShape(0.5f), JPH::RVec3(0.0_r, 3.0_r, 0.0_r), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Physics::Layers::MOVING);
    auto* sphere_ptr = Physics::body_interface->CreateBody(sphere_settings);
    auto sphere_id = sphere_ptr->GetID();

    Physics::body_interface->SetRestitution(sphere_id, .8f);

    Game::ECS::PhysicsDynamic asdf;
    asdf.body = sphere_id;
    Game::registry.emplace<Game::ECS::PhysicsDynamic>(Game::testEntity, asdf);

    Physics::engine.OptimizeBroadPhase();
  }

  void TerminateGame()
  {
    Game::testMesh.vertexBuffer.reset();
    Game::testMesh.indexBuffer.reset();
  }

  void TickGameFixed(double dt)
  {
    for (auto [entity] : Game::registry.view<Game::ECS::PhysicsKinematicAdded>().each())
    {
      const auto& body = Game::registry.get<Game::ECS::PhysicsKinematic>(entity).body;
      Physics::body_interface->AddBody(body, JPH::EActivation::DontActivate);
      Physics::bodyToEntity.emplace(body, entity);
      Game::registry.remove<Game::ECS::PhysicsKinematicAdded>(entity);
    }
    
    for (auto [entity] : Game::registry.view<Game::ECS::PhysicsDynamicAdded>().each())
    {
      const auto& body = Game::registry.get<Game::ECS::PhysicsDynamic>(entity).body;
      Physics::body_interface->AddBody(body, JPH::EActivation::Activate);
      Physics::bodyToEntity.emplace(body, entity);
      Game::registry.remove<Game::ECS::PhysicsDynamicAdded>(entity);
    }
    
    Physics::engine.Update(float(dt), 1, Physics::tempAllocator, Physics::jobSystem);

    auto bodies = JPH::BodyIDVector();
    Physics::engine.GetActiveBodies(JPH::EBodyType::RigidBody, bodies);

    for (const auto& body : bodies)
    {
      if (auto it = Physics::bodyToEntity.find(body); it != Physics::bodyToEntity.end())
      {
        auto entity = it->second;

        if (auto* transform = Game::registry.try_get<Game::ECS::Transform>(entity))
        {
          auto pos = JPH::RVec3();
          auto rot = JPH::Quat();
          Physics::body_interface->GetPositionAndRotation(body, pos, rot);
          transform->position = { pos.GetX(), pos.GetY(), pos.GetZ() };
          transform->rotation = { rot.GetW(), rot.GetX(), rot.GetY(), rot.GetZ() };
        }
      }
    }
  }

  void TickGameVariable([[maybe_unused]] double dt)
  {
    const float dtf = static_cast<float>(dt);
    const glm::vec3 forward = Game::mainCamera.GetForwardDir();
    const glm::vec3 right = glm::normalize(glm::cross(forward, { 0, 1, 0 }));
    if (glfwGetKey(App::window, GLFW_KEY_W) == GLFW_PRESS)
      Game::mainCamera.position += forward * dtf * Game::cameraSpeed;
    if (glfwGetKey(App::window, GLFW_KEY_S) == GLFW_PRESS)
      Game::mainCamera.position -= forward * dtf * Game::cameraSpeed;
    if (glfwGetKey(App::window, GLFW_KEY_D) == GLFW_PRESS)
      Game::mainCamera.position += right * dtf * Game::cameraSpeed;
    if (glfwGetKey(App::window, GLFW_KEY_A) == GLFW_PRESS)
      Game::mainCamera.position -= right * dtf * Game::cameraSpeed;
    if (glfwGetKey(App::window, GLFW_KEY_Q) == GLFW_PRESS)
      Game::mainCamera.position.y -= dtf * Game::cameraSpeed;
    if (glfwGetKey(App::window, GLFW_KEY_E) == GLFW_PRESS)
      Game::mainCamera.position.y += dtf * Game::cameraSpeed;
    Game::mainCamera.yaw += static_cast<float>(App::cursorOffset.x * Game::cursorSensitivity);
    Game::mainCamera.pitch += static_cast<float>(App::cursorOffset.y * Game::cursorSensitivity);
    Game::mainCamera.pitch = glm::clamp(Game::mainCamera.pitch, -glm::half_pi<float>() + 1e-4f, glm::half_pi<float>() - 1e-4f);
  }
}

void MainLoop()
{
  constexpr double GAME_TICK = 1.0 / 60.0;
  double gameTickAccum = 0;
  auto prevTime = glfwGetTime();
  while(!glfwWindowShouldClose(App::window))
  {
    auto curTime = glfwGetTime();
    auto dt = curTime - prevTime;
    prevTime = curTime;

    App::cursorOffset = {};
    glfwPollEvents();

    gameTickAccum += dt;
    if (gameTickAccum >= GAME_TICK)
    {
      gameTickAccum -= GAME_TICK;

      TickGameFixed(GAME_TICK);
    }

    TickGameVariable(dt);
    TickUI(dt);
    TickRender(dt);
  }
}

int main()
{
  InitializeApplication();
  InitializeRenderer();
  InitializePhysics();
  InitializeGame();
  MainLoop();
  TerminateGame();
  TerminatePhysics();
  TerminateRenderer();
  TerminateApplication();
}