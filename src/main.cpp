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
#include <Fwog/Texture.h>

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
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Character/Character.h>
#include <Jolt/Physics/Collision/ContactListener.h>

#include <numeric>
#include <filesystem>
#include <optional>
#include <iostream>
#include <fstream>
#include <exception>
#include <thread>
#include <unordered_map>
#include <functional>

using namespace JPH::literals;

// Globals
namespace
{
  namespace PCG
  {
    constexpr std::uint32_t Hash(std::uint32_t seed)
    {
      std::uint32_t state = seed * 747796405u + 2891336453u;
      std::uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
      return (word >> 22u) ^ word;
    }

    // Used to advance the PCG state.
    constexpr std::uint32_t RandU32(std::uint32_t& rng_state)
    {
      std::uint32_t state = rng_state;
      rng_state           = rng_state * 747796405u + 2891336453u;
      std::uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
      return (word >> 22u) ^ word;
    }

    // Advances the prng state and returns the corresponding random float.
    constexpr float RandFloat(std::uint32_t& state, float min = 0, float max = 1)
    {
      state   = RandU32(state);
      float f = float(state) * std::bit_cast<float>(0x2f800004u);
      return f * (max - min) + min;
    }

    std::uint32_t state{};
  } // namespace PCG

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
      glm::vec3 color{};
    };

    struct Mesh
    {
      std::optional<Fwog::TypedBuffer<Vertex>> vertexBuffer;
      std::optional<Fwog::TypedBuffer<index_t>> indexBuffer;
      std::vector<Vertex> vertices;
      std::vector<index_t> indices;
    };

    struct InstanceUniforms
    {
      glm::mat4 world_from_object;
      glm::vec4 tintColor = glm::vec4(1);
    };

    struct FrameUniforms
    {
      glm::mat4 clip_from_world{};
      glm::mat4 light_from_world{};
      glm::vec3 sunDirection = glm::normalize(glm::vec3{.2f, -1.f, 0});
      uint32_t padding{};
      glm::vec3 camPosition{};
    };

    std::optional<Fwog::GraphicsPipeline> pipeline;
    std::optional<Fwog::GraphicsPipeline> shadowPipeline;
    std::optional<Fwog::TypedBuffer<InstanceUniforms>> instanceBuffer;
    std::optional<Fwog::TypedBuffer<FrameUniforms>> frameUniformsBuffer;
    std::optional<Fwog::Texture> shadowMap;
    FrameUniforms frameUniforms{};

    // Shaders
    namespace
    {
      const char* sceneVertexSource = R"(
#version 460 core

struct Vertex
{
  float px, py, pz;
  float nx, ny, nz;
  float cr, cg, cb;
};

struct InstanceUniforms
{
  mat4 world_from_object;
  vec4 tintColor;
};

layout(binding = 0, std430) readonly buffer VertexBuffer
{
  Vertex vertices[];
};

layout(binding = 1, std430) readonly buffer FrameUniforms
{
  mat4 clip_from_world;
  mat4 light_from_world;
  vec3 sunDirection;
  vec3 camPosition;
}frame;

layout(binding = 2, std430) readonly buffer InstanceBuffer
{
  InstanceUniforms instances[];
};

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_color;
layout(location = 3) out vec4 v_tint;

void main()
{
  Vertex v = vertices[gl_VertexID];

  const vec3 posObj = vec3(v.px, v.py, v.pz);
  const vec3 normObj = vec3(v.nx, v.ny, v.nz);

  InstanceUniforms instance = instances[gl_InstanceID + gl_BaseInstance];

  v_position = (instance.world_from_object * vec4(posObj, 1.0)).xyz;
  v_normal   = transpose(inverse(mat3(instance.world_from_object))) * normObj;
  v_color    = vec3(v.cr, v.cg, v.cb);
  v_tint     = instance.tintColor;

  gl_Position = frame.clip_from_world * vec4(v_position, 1.0);
}
)";

      const char* sceneFragmentSource = R"(
#version 460 core

layout(binding = 1, std430) readonly buffer FrameUniforms
{
  mat4 clip_from_world;
  mat4 light_from_world;
  vec3 sunDirection;
  vec3 camPosition;
}frame;

layout(binding = 0) uniform sampler2DShadow s_shadowMap;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_color;
layout(location = 3) in vec4 v_tint;

layout(location = 0) out vec4 o_color;

void main()
{
  // Shadow
  const vec4 lightClip = frame.light_from_world * vec4(v_position, 1);
  const vec3 lightNdc = lightClip.xyz / lightClip.w;
  const vec3 lightUv = lightNdc * 0.5 + 0.5;
  const float shadow = textureLod(s_shadowMap, lightUv, 0);
  const float sunDiffuse = shadow * max(0, dot(v_normal, -frame.sunDirection));

  vec3 specular = vec3(0);
  if (v_tint.a > 1) // Hacky way to get phong-like specular
  {
    const vec3 surfaceToCam = normalize(frame.camPosition - v_position);
    const vec3 reflected = normalize(reflect(frame.sunDirection, v_normal));
    const float spec = pow(max(0, dot(reflected, surfaceToCam)), v_tint.a);
    //specular = spec * v_tint.rgb;
    specular = vec3(spec);
  }

  const vec3 albedo = v_tint.rgb * mix(v_color, v_normal * .5 + .5, .1);
  o_color = vec4(specular + 0.8 * sunDiffuse * albedo + 0.2 * albedo, 1.0);
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
        const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override
        {
          switch ((JPH::BroadPhaseLayer::Type)inLayer)
          {
          case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING:	return "NON_MOVING";
          case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::MOVING:		return "MOVING";
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
    //JPH::CharacterVirtual* character{};
    JPH::Character* character{};
  }

  namespace Game
  {
    namespace ECS
    {
      struct Frog {};
      struct Dung {};
      struct Floor {};
      struct DeferDelete{};

      struct Transform
      {
        glm::vec3 position = { 0, 0, 0 };
        glm::quat rotation = { 1, 0, 0, 0 };
        glm::vec3 scale = { 1, 1, 1 };
      };

      struct PreviousModel
      {
        glm::mat4 world_from_object = glm::translate(glm::mat4(1), glm::vec3(100000)); // Epic hack to prevent frame of flickering when spawning objects
      };

      struct MeshRef
      {
        Render::Mesh* mesh{};
      };

      struct Tint
      {
        glm::vec4 color = {1, 1, 1, 1};
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

    enum class State
    {
      MAIN_MENU,
      GAME,
      PAUSE,
      DIED,
    };

    State state     = State::MAIN_MENU;
    State lastState = State::MAIN_MENU;

    void SetState(State s)
    {
      lastState = state;
      state     = s;
    }

    struct DifficultyConfig
    {
      double timeBetweenDungDrops{};
      double timeBetweenOppDrops{};
      int numOppsPerDrop = 1;
      int numDungPerDrop = 1;
      float velocityFactor = 1;
    };

    View mainCamera{};
    float cursorSensitivity            = 0.25f;
    float fovyDeg                      = 60.0f;
    float playerAcceleration           = 12;
    float playerMaxSpeed               = 2.5f;
    float playerInitialJumpVelocity    = 3.5f;  // Vertical speed immediately after pressing jump
    float playerJumpAcceleration       = 10;   // Vertical acceleration when holding jump
    float playerTimeSinceOnGround      = 1000;
    float playerJumpModulationDuration = 0.2f; // Longer duration means more velocity can be added if the player holds jump
    float playerFriction               = 1.2f;
    int difficulty                     = 1;
    int maxDifficulty                  = 5; // Highest difficulty attained, play more to unlock more
    int maxDifficultyPlayed            = maxDifficulty; // Difficulties above this show as green in the menu
    double secondsSurvived             = 0; // Current run duration
    double timeSinceDied               = 0;
    int moneyCollected                 = 0;
    entt::entity floorEntity           = entt::null;
    bool playerHasCamoufroge           = false;
    double timeSinceSpawnedOpps        = 0;
    double timeSinceSpawnedDung        = 0;
    int irsDonations                   = 0;

    constexpr glm::vec4 terrainDefaultColor = {.2, .5, .1, 0};
    constexpr glm::vec4 terrainFrockeyColor = {.9, .9, .96, 15};
    float terrainDefaultFriction            = 0.2f;
    float terrainFrockeyFriction            = 0.0f;

    struct ShopItem
    {
      bool owned       = false;
      int cost         = 0;
      const char* name = "sample text";
      const char* tooltip{};
      std::function<void(void)> onPurchase;
      std::function<void(void)> onRefund;
    };

    // Shopping
    namespace
    {
      struct ShopRow
      {
        std::vector<ShopItem> items;
      };

      std::vector<ShopRow> shop;
    } // namespace

    // Game objects
    entt::registry registry;

    Render::Mesh cubeMesh;
    Render::Mesh sphereMesh;
    Render::Mesh frogMesh;
    Render::Mesh envMesh;

    const char* deathMessages[] = {
      "You got frogged!",
      "You became a frog's meal!",
      "You were food, not friend.",
      "RIP bozo",
      "Caught!",
      "Froggylicious! (you were eaten)",
      "...",
      "You croaked.",
      "You were toadally delicious.",
      "RIF (rest in frog)",
      "<ribbit>",
      "You forgot to bring a croaking device.",
      "You came to a ribbeting demise.",
      "Frogs love hip-hop... and you.",
    };

    int deathMessageIndex = 0;
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
    glfwWindowHint(GLFW_SAMPLES, 4); // MSAA x4

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor)
    {
      throw std::runtime_error("Failed to find monitor");
    }
    const GLFWvidmode* videoMode = glfwGetVideoMode(monitor);
    // TODO: update app title
    App::window = glfwCreateWindow(static_cast<int>(videoMode->width * .75), static_cast<int>(videoMode->height * .75), "Afrocalypse", nullptr, nullptr);
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
    // TODO: proper GUI scaling
    //float xscale, yscale;
    //glfwGetWindowContentScale(App::window, &xscale, &yscale);
    //const auto contentScale = std::max(xscale, yscale);
    //const float fontSize    = glm::floor(18 * contentScale);
    //ImGui::GetStyle().ScaleAllSizes(contentScale);
    ImGui_ImplGlfw_InitForOpenGL(App::window, true);
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();

    glfwSetInputMode(App::window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
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

    auto mesh = Render::Mesh{};

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

          //// Check if `texcoord_index` is zero or positive. negative = no texcoord data
          //if (idx.texcoord_index >= 0)
          //{
          //  tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
          //  tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
          //  vertex.texcoord = { tx, ty };
          //}

          // Optional: vertex colors
          tinyobj::real_t red   = attrib.colors[3 * size_t(idx.vertex_index) + 0];
          tinyobj::real_t green = attrib.colors[3 * size_t(idx.vertex_index) + 1];
          tinyobj::real_t blue  = attrib.colors[3 * size_t(idx.vertex_index) + 2];
          vertex.color = { red, green, blue };

          mesh.vertices.push_back(vertex);
        }
        index_offset += fv;
      }
    }

    mesh.indices = std::vector<Render::index_t>(mesh.vertices.size());
    std::iota(mesh.indices.begin(), mesh.indices.end(), 0);
    
    mesh.indexBuffer.emplace(mesh.indices);
    mesh.vertexBuffer.emplace(mesh.vertices);
    return mesh;
  }

  void InitializeRenderer()
  {
    auto vertexShader   = Fwog::Shader(Fwog::PipelineStage::VERTEX_SHADER, Render::sceneVertexSource, "Scene vertex shader");
    auto fragmentShader = Fwog::Shader(Fwog::PipelineStage::FRAGMENT_SHADER, Render::sceneFragmentSource, "Scene fragment shader");
    Render::pipeline    = Fwog::GraphicsPipeline({
         .name           = "Scene pipeline",
         .vertexShader   = &vertexShader,
         .fragmentShader = &fragmentShader,
         .rasterizationState =
        {
             .cullMode = Fwog::CullMode::BACK,
        },
         .depthState =
        {
             .depthTestEnable  = true,
             .depthWriteEnable = true,
        },
    });

    Render::frameUniformsBuffer.emplace(Fwog::BufferStorageFlag::DYNAMIC_STORAGE);
    Render::shadowMap.emplace(Fwog::CreateTexture2D({2048, 2048}, Fwog::Format::D24_UNORM, "Shadow map"));
    Render::shadowPipeline = Fwog::GraphicsPipeline({
      .name         = "Shadow pipeline",
      .vertexShader = &vertexShader,
      .rasterizationState =
        {
          .cullMode = Fwog::CullMode::NONE,
          .depthBiasEnable = true,
          .depthBiasConstantFactor = 500,
          .depthBiasSlopeFactor = 5,
        },
      .depthState =
        {
          .depthTestEnable  = true,
          .depthWriteEnable = true,
        },
    });
  }

  void TerminateRenderer()
  {
    Render::shadowPipeline.reset();
    Render::shadowMap.reset();
    Render::instanceBuffer.reset();
    Render::frameUniformsBuffer.reset();
    Render::pipeline.reset();
    Fwog::Terminate();
  }

  void TickRender([[maybe_unused]] double dt, float interpolant)
  {
    static auto lastInterpolant = interpolant;
    auto renderMarker = Fwog::ScopedDebugMarker("TickRender");

    glEnable(GL_FRAMEBUFFER_SRGB);

    struct Instance
    {
      size_t index;
      Render::Mesh* mesh;
    };

    auto instanceData = std::vector<Render::InstanceUniforms>();
    auto instances    = std::vector<Instance>();

    auto view = Game::registry.view<Game::ECS::Transform, Game::ECS::MeshRef>();
    for (auto&& [entity, transform, meshRef] : view.each())
    {
      instances.emplace_back(instanceData.size(), meshRef.mesh);

      auto world_from_object = glm::translate(glm::identity<glm::mat4>(), transform.position) * glm::mat4_cast(transform.rotation) *
                               glm::scale(glm::identity<glm::mat4>(), transform.scale);

      auto renderModel = world_from_object;
      if (auto* p = Game::registry.try_get<Game::ECS::PreviousModel>(entity))
      {
        //renderModel = glm::mix(p->world_from_object, world_from_object, interpolant);
        for (int i = 0; i < renderModel.length(); i++)
        {
          renderModel[i] = glm::mix(p->world_from_object[i], world_from_object[i], interpolant);
        }

        // Reset matrix
        if (lastInterpolant > interpolant)
        {
          p->world_from_object = world_from_object;
        }
      }

      auto tintColor = glm::vec4(1);
      if (auto* p = Game::registry.try_get<Game::ECS::Tint>(entity))
      {
        tintColor = p->color;
      }
      instanceData.emplace_back(renderModel, tintColor);
    }

    lastInterpolant = interpolant;

    if (!Render::instanceBuffer || Render::instanceBuffer->Size() < instances.size() * sizeof(Render::InstanceUniforms))
    {
      Render::instanceBuffer.emplace(instanceData.size() * 3 / 2, Fwog::BufferStorageFlag::DYNAMIC_STORAGE, "Instance Data");
    }

    if (!instanceData.empty())
    {
      Render::instanceBuffer->UpdateData(instanceData);
    }

    const auto lightView_from_world = glm::lookAt(glm::vec3(0), Render::frameUniforms.sunDirection, glm::vec3(0, 1, 0));
    Render::frameUniforms.light_from_world = glm::ortho(-30.0f, 30.0f, -30.0f, 30.0f, -200.0f, 200.0f) * lightView_from_world;
    Render::frameUniforms.camPosition      = Game::mainCamera.position;

    Fwog::Render(
      Fwog::RenderInfo{
        .name = "Shadow pass",
        .depthAttachment =
          Fwog::RenderDepthStencilAttachment{
            .texture    = Render::shadowMap.value(),
            .loadOp     = Fwog::AttachmentLoadOp::CLEAR,
            .clearValue = {.depth = 1},
          },
      },
      [&]
      {
        Render::frameUniforms.clip_from_world = Render::frameUniforms.light_from_world;
        Render::frameUniformsBuffer->UpdateData(Render::frameUniforms);

        Fwog::Cmd::BindGraphicsPipeline(Render::shadowPipeline.value());
        Fwog::Cmd::BindStorageBuffer(1, Render::frameUniformsBuffer.value());
        Fwog::Cmd::BindStorageBuffer(2, Render::instanceBuffer.value());

        for (size_t i = 0; i < instances.size(); i++)
        {
          const auto& instance = instances[i];
          Fwog::Cmd::BindIndexBuffer(instance.mesh->indexBuffer.value(), Fwog::IndexType::UNSIGNED_INT);
          Fwog::Cmd::BindStorageBuffer(0, instance.mesh->vertexBuffer.value());
          Fwog::Cmd::DrawIndexed((uint32_t)instance.mesh->indices.size(), 1, 0, 0, (uint32_t)i);
        }
      });

    Fwog::RenderToSwapchain(
      Fwog::SwapchainRenderInfo{
        .name            = "Scene pass",
        .viewport        = Fwog::Viewport{.drawRect{.offset = {0, 0}, .extent = {(uint32_t)App::windowSize.x, (uint32_t)App::windowSize.y}}},
        .colorLoadOp     = Fwog::AttachmentLoadOp::CLEAR,
        .clearColorValue = {.3f, .4f, .75f, 1.0f},
        .depthLoadOp     = Fwog::AttachmentLoadOp::CLEAR,
        .clearDepthValue = 1.0f,
      },
      [&]
      {
        auto projection = glm::perspective(glm::radians(Game::fovyDeg), (float)App::windowSize.x / (float)App::windowSize.y, 0.1f, 200.0f);
        Render::frameUniforms.clip_from_world = projection * Game::mainCamera.GetViewMatrix();
        Render::frameUniformsBuffer->UpdateData(Render::frameUniforms);

        Fwog::Cmd::BindGraphicsPipeline(Render::pipeline.value());
        Fwog::Cmd::BindStorageBuffer(1, Render::frameUniformsBuffer.value());
        Fwog::Cmd::BindStorageBuffer(2, Render::instanceBuffer.value());
        Fwog::Cmd::BindSampledImage(0,
          Render::shadowMap.value(),
          Fwog::Sampler({
            .minFilter    = Fwog::Filter::LINEAR,
            .magFilter    = Fwog::Filter::LINEAR,
            .addressModeU = Fwog::AddressMode::CLAMP_TO_EDGE,
            .addressModeV = Fwog::AddressMode::CLAMP_TO_EDGE,
            .compareEnable = true,
            .compareOp     = Fwog::CompareOp::LESS,
          }));

        for (size_t i = 0; i < instances.size(); i++)
        {
          const auto& instance = instances[i];
          Fwog::Cmd::BindIndexBuffer(instance.mesh->indexBuffer.value(), Fwog::IndexType::UNSIGNED_INT);
          Fwog::Cmd::BindStorageBuffer(0, instance.mesh->vertexBuffer.value());
          Fwog::Cmd::DrawIndexed((uint32_t)instance.mesh->indices.size(), 1, 0, 0, (uint32_t)i);
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

  void ShowShopInterface();
  void BeginRound();

  void TickUI([[maybe_unused]] double dt)
  {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    switch (Game::state)
    {
    case Game::State::MAIN_MENU:
    {
      glfwSetInputMode(App::window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
      ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
      ImGui::SetNextWindowSize(ImVec2(370, 420));
      if (ImGui::Begin("common", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration))
      {
        // Show difficulty selectors
        ImGui::SeparatorText("Afrocalypse");
        ImGui::TextUnformatted("Select a difficulty to start");
        for (int i = 1; i <= 11; i++)
        {
          const auto disabled = i > Game::maxDifficulty;
          const auto pushButtonColor = !disabled && i > Game::maxDifficultyPlayed;
          ImGui::BeginDisabled(disabled);
          if (pushButtonColor)
          {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.3f, 1));
          }
          if (ImGui::Button(std::to_string(i).c_str()))
          {
            Game::maxDifficultyPlayed = glm::max(Game::maxDifficultyPlayed, i);
            Game::difficulty = i;
            Game::SetState(Game::State::GAME);
            BeginRound();
          }
          if (pushButtonColor)
          {
            ImGui::PopStyleColor();
          }
          ImGui::EndDisabled();
          if (i != 11)
          {
            ImGui::SameLine();
          }
        }

        // Show shop
        ImGui::NewLine();
        ShowShopInterface();
        ImGui::NewLine();

        // Options
        ImGui::SeparatorText("Options");
        ImGui::SliderFloat("Cursor speed", &Game::cursorSensitivity, 0, 2, "%.2f");
        ImGui::SliderFloat("FoV", &Game::fovyDeg, 30, 90, "%.0f");
        ImGui::NewLine();

        if (ImGui::Button("Quit to desktop"))
        {
          glfwSetWindowShouldClose(App::window, GLFW_TRUE);
        }
      }
      ImGui::End();
      break;
    }
    case Game::State::GAME:
    {
      glfwSetInputMode(App::window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
      // TODO:: show money and time

      ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.15f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
      ImGui::SetNextWindowSize(ImVec2(160, 90));
      if (ImGui::Begin("common", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground))
      {
        ImGui::Text("Difficulty: %d", Game::difficulty);
        ImGui::Text("Survived:   %.0f seconds", glm::floor(Game::secondsSurvived));
        ImGui::Text("Currency:   %d D", Game::moneyCollected);
      }
      ImGui::End();
      
      break;
    }
    case Game::State::PAUSE:
    {
      glfwSetInputMode(App::window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

      ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
      ImGui::SetNextWindowSize(ImVec2(160, 90));
      if (ImGui::Begin("common", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration))
      {
        if (ImGui::Button("Resume"))
        {
          Game::SetState(Game::State::GAME);
        }
        //ImGui::NewLine();
        //// Options
        //ImGui::SliderFloat("Cursor speed", &Game::cursorSensitivity, 0, 2, "%.2f");
        //ImGui::SliderFloat("FoV", &Game::fovyDeg, 30, 90, "%.0f");
        //ImGui::NewLine();

        if (ImGui::Button("Exit to main menu"))
        {
          Game::SetState(Game::State::MAIN_MENU);
        }
        if (ImGui::Button("Quit to desktop"))
        {
          glfwSetWindowShouldClose(App::window, GLFW_TRUE);
        }
      }
      ImGui::End();
      break;
    }
    case Game::State::DIED:
    {
      ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
      ImGui::SetNextWindowSize(ImVec2(320, 50));
      if (ImGui::Begin("common", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration))
      {
        ImGui::TextUnformatted(Game::deathMessages[Game::deathMessageIndex]);
      }
      ImGui::End();
      break;
    }
    default:;
    }
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

  auto CreateMeshShape(const Render::Mesh& mesh)
  {
    auto vertices  = JPH::Array<JPH::Float3>();
    auto triangles = JPH::Array<JPH::IndexedTriangle>();

    for (const auto& v : mesh.vertices)
    {
      vertices.push_back(JPH::Float3(v.position.x, v.position.y, v.position.z));
    }

    assert(mesh.indices.size() % 3 == 0);
    for (size_t i = 0; i < mesh.indices.size(); i += 3)
    {
      triangles.push_back(JPH::IndexedTriangle(mesh.indices[i + 0], mesh.indices[i + 1], mesh.indices[i + 2]));
    }

    auto meshShapeSettings = JPH::MeshShapeSettings(JPH::VertexList(vertices), JPH::IndexedTriangleList(triangles));
    meshShapeSettings.SetEmbedded();
    auto meshShapeResult = meshShapeSettings.Create();
    auto meshShape       = meshShapeResult.Get();
    meshShape->SetEmbedded();
    return meshShape;
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
    
    constexpr JPH::uint cMaxBodies = 5000;
    constexpr JPH::uint cNumBodyMutexes = 0;
    constexpr JPH::uint cMaxBodyPairs = 5000;
    constexpr JPH::uint cMaxContactConstraints = 5000;
    Physics::engine.Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints, Physics::broad_phase_layer_interface, Physics::object_vs_broadphase_layer_filter, Physics::object_vs_object_layer_filter);

    Physics::body_interface = &Physics::engine.GetBodyInterface();
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
  entt::entity SpawnRegularFrog(glm::vec3 pos, glm::vec3 velocity = {})
  {
    auto sphereSettings = JPH::BodyCreationSettings(new JPH::SphereShape(0.5f), JPH::RVec3(pos.x, pos.y, pos.z), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Physics::Layers::MOVING);
    auto* spherePtr = Physics::body_interface->CreateBody(sphereSettings);
    auto sphereID  = spherePtr->GetID();
    Physics::body_interface->SetRestitution(sphereID, .8f);
    Physics::body_interface->AddBody(sphereID, JPH::EActivation::Activate);
    Physics::body_interface->SetLinearVelocity(sphereID, JPH::Vec3Arg(velocity.x, velocity.y, velocity.z));

    auto entity = Game::registry.create();
    Game::registry.emplace<Game::ECS::Frog>(entity);
    auto& transform = Game::registry.emplace<Game::ECS::Transform>(entity);
    
    transform.scale = { 0.5f, 0.5f, 0.5f };

    Game::registry.emplace<Game::ECS::MeshRef>(entity).mesh = &Game::frogMesh;
    Game::registry.emplace<Game::ECS::PhysicsDynamic>(entity).body = sphereID;
    Game::registry.emplace<Game::ECS::PreviousModel>(entity);
    Game::registry.emplace<Game::ECS::Tint>(entity).color = {1, 1, 1, 1};

    return entity;
  }

  entt::entity SpawnDung(glm::vec3 pos, glm::vec3 velocity = {})
  {
    auto sphereSettings = JPH::BodyCreationSettings(new JPH::SphereShape(0.3f), JPH::RVec3(pos.x, pos.y, pos.z), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Physics::Layers::MOVING);
    auto* spherePtr = Physics::body_interface->CreateBody(sphereSettings);
    auto sphereID  = spherePtr->GetID();
    Physics::body_interface->SetRestitution(sphereID, .2f);
    Physics::body_interface->AddBody(sphereID, JPH::EActivation::Activate);
    Physics::body_interface->SetLinearVelocity(sphereID, JPH::Vec3Arg(velocity.x, velocity.y, velocity.z));

    auto entity = Game::registry.create();
    Game::registry.emplace<Game::ECS::Dung>(entity);
    auto& transform = Game::registry.emplace<Game::ECS::Transform>(entity);
    
    transform.scale = { 0.3f, 0.3f, 0.3f };

    Game::registry.emplace<Game::ECS::MeshRef>(entity).mesh = &Game::sphereMesh;
    Game::registry.emplace<Game::ECS::PhysicsDynamic>(entity).body = sphereID;
    Game::registry.emplace<Game::ECS::PreviousModel>(entity);
    Game::registry.emplace<Game::ECS::Tint>(entity).color = {0.844f, 0.676f, 0.273f, 15};

    return entity;
  }

  void BeginRound()
  {
    Game::mainCamera.position = {0, 2, 0};
    Physics::character->SetPosition(JPH::Vec3Arg(0, 2, 0));
    Physics::character->SetLinearVelocity(JPH::Vec3Arg(0, 0, 0));
    Game::secondsSurvived = 0;
    Game::timeSinceDied   = 0;
    Game::timeSinceSpawnedOpps = -2; // Give the player a moment before the afrocalypse starts
    Game::timeSinceSpawnedDung = 0;

    auto frogs = Game::registry.view<Game::ECS::Frog>();
    Game::registry.destroy(frogs.begin(), frogs.end());

    auto dung = Game::registry.view<Game::ECS::Dung>();
    Game::registry.destroy(dung.begin(), dung.end());
  }

  Game::DifficultyConfig GetDifficultyConfig([[maybe_unused]] int difficulty)
  {
    // Config for difficulty 1
    Game::DifficultyConfig config{
      .timeBetweenDungDrops = 1.2f - 0.1f * (difficulty - 1),
      .timeBetweenOppDrops  = 2.5f - 0.2f * (difficulty -1),
      .numOppsPerDrop = 1,
      .numDungPerDrop = 1,
      .velocityFactor       = 0.8f + 0.04f * (difficulty - 1),
    };

    if (difficulty >= 6)
    {
      config.numDungPerDrop++;
      config.numOppsPerDrop++;
    }

    if (difficulty >= 10)
    {
      config.numDungPerDrop++;
      config.numOppsPerDrop++;
    }

    if (difficulty >= 11)
    {
      config.numDungPerDrop++;
      config.numOppsPerDrop++;
    }

    return config;
  }

  void PopulateShop()
  {
    Game::ShopRow speedRow;
    for (int i = 0; i < 5; i++)
    {
      speedRow.items.push_back(Game::ShopItem{
        .cost       = 5 + 10 * i,
        .name       = "Budgett's walking stick",
        .tooltip    = "Increase max speed by 20%",
        .onPurchase = [] { Game::playerMaxSpeed *= 1.2f; },
        .onRefund   = [] { Game::playerMaxSpeed /= 1.2f; },
      });
    }
    Game::shop.push_back(speedRow);

    Game::ShopRow accelRow;
    for (int i = 0; i < 5; i++)
    {
      accelRow.items.push_back(Game::ShopItem{
        .cost       = 4 + 4 * i,
        .name       = "Wednesday siren",
        .tooltip    = "Increase acceleration by 30%",
        .onPurchase = [] { Game::playerAcceleration *= 1.3f; },
        .onRefund   = [] { Game::playerAcceleration /= 1.3f; },
      });
    }
    Game::shop.push_back(accelRow);

    Game::ShopRow jumpRow;
    for (int i = 0; i < 5; i++)
    {
      jumpRow.items.push_back(Game::ShopItem{
        .cost       = 10 + 5 * i,
        .name       = "Surgically grafted frog legs",
        .tooltip    = "Increase base jump velocity by 10%",
        .onPurchase = [] { Game::playerInitialJumpVelocity *= 1.1f; },
        .onRefund   = [] { Game::playerInitialJumpVelocity /= 1.1f; },
      });
    }
    Game::shop.push_back(jumpRow);

    Game::ShopRow camoRow;
    camoRow.items.push_back(Game::ShopItem{
      .cost = 50,
      .name = "Camoufroge",
      .tooltip = "Gives you a 50% chance to not be eaten when frogged",
      .onPurchase =
        []
      {
        Game::playerHasCamoufroge = true;
      },
      .onRefund =
        []
      {
        Game::playerHasCamoufroge = false;
      },
    });
    Game::shop.push_back(camoRow);

    Game::ShopRow frockeyRow;
    frockeyRow.items.push_back(Game::ShopItem{
      .cost    = -25,
      .name    = "Frockey",
      .tooltip = ":)",
      .onPurchase =
        []
      {
        Game::registry.get<Game::ECS::Tint>(Game::floorEntity).color = Game::terrainFrockeyColor;
        Physics::body_interface->SetFriction(Game::registry.get<Game::ECS::PhysicsKinematic>(Game::floorEntity).body, Game::terrainFrockeyFriction);
      },
      .onRefund =
        []
      {
        Game::registry.get<Game::ECS::Tint>(Game::floorEntity).color = Game::terrainDefaultColor;
        Physics::body_interface->SetFriction(Game::registry.get<Game::ECS::PhysicsKinematic>(Game::floorEntity).body, Game::terrainDefaultFriction);
      },
    });
    Game::shop.push_back(frockeyRow);
  }

  void ShowShopInterface()
  {
    ImGui::SeparatorText("Bugge Shoppe");
    ImGui::Text("Currency: %d D", Game::moneyCollected);
    if (ImGui::Button("Refund all"))
    {
      for (auto& row : Game::shop)
      {
        for (auto& item : row.items)
        {
          if (item.owned)
          {
            item.onRefund();
            Game::moneyCollected += item.cost;
            item.owned = false;
          }
        }
      }
    }
    for (auto j = 0; auto& row : Game::shop)
    {
      ImGui::PushID(j++);
      bool prevItemIsntOwned = false;
      for (size_t i = 0; i < row.items.size(); i++)
      {
        auto& item = row.items[i];

        // Only show name for first item in a row
        if (i == 0)
        {
          ImGui::TextUnformatted(item.name);
          if (item.tooltip)
          {
            ImGui::SetItemTooltip("%s", item.tooltip);
          }
          ImGui::SameLine();
        }

        ImGui::PushID((int)i);
        
        ImGui::BeginDisabled(prevItemIsntOwned || Game::moneyCollected < item.cost || item.owned);
        const bool itemOwned = item.owned;
        if (itemOwned)
        {
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 1));
        }
        if (ImGui::Button(std::to_string(item.cost).c_str()))
        {
          item.onPurchase();
          Game::moneyCollected -= item.cost;
          item.owned = true;
        }
        if (itemOwned)
        {
          ImGui::PopStyleColor();
        }
        ImGui::EndDisabled();

        ImGui::PopID();


        if (i != row.items.size() - 1)
        {
          ImGui::SameLine();
        }

        prevItemIsntOwned = !item.owned;
      }
      ImGui::PopID();
    }
    ImGui::TextUnformatted("IRS donation");
    ImGui::SameLine();
    if (ImGui::Button("10##asdf"))
    {
      Game::moneyCollected -= 10;
      Game::irsDonations++;
    }
  }

  void InitializeGame()
  {
    PCG::state = PCG::Hash((uint32_t)std::time(nullptr));

    PopulateShop();
    Game::cubeMesh            = LoadObjFile(GetDataDirectory() / "models/cube.obj");
    Game::sphereMesh          = LoadObjFile(GetDataDirectory() / "models/sphere.obj");
    Game::frogMesh            = LoadObjFile(GetDataDirectory() / "models/frog.obj");
    Game::envMesh             = LoadObjFile(GetDataDirectory() / "models/ground.obj");

    // Add static floor
    auto floorShape = CreateMeshShape(Game::envMesh);
    auto floor_settings = JPH::BodyCreationSettings(floorShape, JPH::RVec3(0.0_r, 0.0_r, 0.0_r), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Physics::Layers::NON_MOVING);
    auto* floor = Physics::body_interface->CreateBody(floor_settings);
    floor->SetRestitution(.0f);
    floor->SetFriction(Game::terrainDefaultFriction);

    Game::floorEntity                                                     = Game::registry.create();
    Game::registry.emplace<Game::ECS::Transform>(Game::floorEntity).scale = {1, 1, 1};
    Game::registry.emplace<Game::ECS::MeshRef>(Game::floorEntity).mesh    = &Game::envMesh;
    Game::registry.emplace<Game::ECS::Floor>(Game::floorEntity);
    Game::registry.emplace<Game::ECS::PhysicsKinematic>(Game::floorEntity).body = floor->GetID();
    Game::registry.emplace<Game::ECS::Tint>(Game::floorEntity).color            = Game::terrainDefaultColor;

    // Add player character
    auto characterSettings = JPH::CharacterSettings();
    characterSettings.SetEmbedded();
    characterSettings.mLayer = Physics::Layers::MOVING;
    characterSettings.mShape = new JPH::SphereShape(0.5f);
    characterSettings.mFriction = Game::playerFriction;
    Physics::character = new JPH::Character(&characterSettings, JPH::Vec3Arg(0, 2, 0), JPH::Quat::sIdentity(), 0, &Physics::engine);
    Physics::body_interface->SetRestitution(Physics::character->GetBodyID(), 0);
    Physics::character->AddToPhysicsSystem();
    Physics::character->SetShape(new JPH::CapsuleShape(0.5f, 0.2f), FLT_MAX);

    struct ContactListenerImpl : JPH::ContactListener
    {
      ContactListenerImpl() = default;

      void OnContactAdded(const JPH::Body& inBody1,
        const JPH::Body& inBody2,
        [[maybe_unused]] const JPH::ContactManifold& inManifold,
        [[maybe_unused]] JPH::ContactSettings& ioSettings) override
      {
        auto ee1 = Physics::bodyToEntity.find(inBody1.GetID());
        auto ee2 = Physics::bodyToEntity.find(inBody2.GetID());

        entt::entity e1 = ee1 != Physics::bodyToEntity.end() ? ee1->second : entt::null;
        entt::entity e2 = ee2 != Physics::bodyToEntity.end() ? ee2->second : entt::null;

        entt::entity frog = entt::null;
        entt::entity dung = entt::null;

        if (e1 != entt::null && Game::registry.any_of<Game::ECS::Frog>(e1))
        {
          frog = e1;
        }

        if (e2 != entt::null && Game::registry.any_of<Game::ECS::Frog>(e2))
        {
          frog = e2;
        }

        if (e1 != entt::null && Game::registry.any_of<Game::ECS::Dung>(e1))
        {
          dung = e1;
        }

        if (e2 != entt::null && Game::registry.any_of<Game::ECS::Dung>(e2))
        {
          dung = e2;
        }

        // Check if the player collided with event thingy
        if (inBody1.GetID() == Physics::character->GetBodyID() || inBody2.GetID() == Physics::character->GetBodyID())
        {
          if (frog != entt::null)
          {
            if ((!Game::playerHasCamoufroge || PCG::RandU32(PCG::state) % 2 == 0) && Game::timeSinceDied <= 0)
            {
              // TODO: play death sound
              Game::SetState(Game::State::DIED);
              Game::deathMessageIndex = PCG::RandU32(PCG::state) % std::size(Game::deathMessages);
            }
          }

          if (dung != entt::null)
          {
            Game::moneyCollected++;
            Game::registry.emplace<Game::ECS::DeferDelete>(dung);
          }
        }
        else if (frog != entt::null)
        {
          // TODO: play frog collision sound
        }
      }
    };
    Physics::engine.SetContactListener(new ContactListenerImpl());

    Physics::engine.OptimizeBroadPhase();
  }

  void TerminateGame()
  {
    Game::cubeMesh.vertexBuffer.reset();
    Game::cubeMesh.indexBuffer.reset();

    Game::sphereMesh.vertexBuffer.reset();
    Game::sphereMesh.indexBuffer.reset();

    Game::frogMesh.vertexBuffer.reset();
    Game::frogMesh.indexBuffer.reset();

    Game::envMesh.vertexBuffer.reset();
    Game::envMesh.indexBuffer.reset();
  }

  void TickGameFixed(double dt)
  {
    if (Game::state != Game::State::GAME && Game::state != Game::State::DIED)
    {
      return;
    }

    // Things still render and simulate, but the camera remains locked in place
    if (Game::state == Game::State::DIED)
    {
      Game::timeSinceDied += dt;
      if (Game::timeSinceDied > 3)
      {
        Game::SetState(Game::State::MAIN_MENU);
      }
    }

    if (auto pos = Physics::character->GetPosition(); pos.GetY() < -100 && Game::timeSinceDied <= 0)
    {
      pos.SetY(0);
      Physics::character->SetPosition(pos);
      Game::SetState(Game::State::DIED);
      Game::deathMessageIndex = PCG::RandU32(PCG::state) % std::size(Game::deathMessages);
    }
    
    auto deleted = Game::registry.view<Game::ECS::DeferDelete>();
    Game::registry.destroy(deleted.begin(), deleted.end());

    // Fling frogs and dung randomly from around the map
    Game::timeSinceSpawnedDung += dt;
    Game::timeSinceSpawnedOpps += dt;
    auto config = GetDifficultyConfig(Game::difficulty);
    if (Game::timeSinceSpawnedDung > config.timeBetweenDungDrops)
    {
      Game::timeSinceSpawnedDung -= config.timeBetweenDungDrops;
      for (int i = 0; i < config.numDungPerDrop; i++)
      {
        const auto phi = PCG::RandFloat(PCG::state, 0, glm::two_pi<float>());
        const float r  = 30;
        const auto pos = glm::vec3(glm::cos(phi) * r, 6, glm::sin(phi) * r);
        const auto vel = 20.0f * glm::normalize(glm::vec3(0, 15 + PCG::RandFloat(PCG::state) * 10, 0) - pos); // Shoot at origin
        SpawnDung(pos, vel * config.velocityFactor);
      }
    }

    if (Game::timeSinceSpawnedOpps > config.timeBetweenOppDrops)
    {
      Game::timeSinceSpawnedOpps -= config.timeBetweenOppDrops;
      for (int i = 0; i < config.numOppsPerDrop; i++)
      {
        const auto phi = PCG::RandFloat(PCG::state, 0, glm::two_pi<float>());
        const float r  = 30;
        const auto pos = glm::vec3(glm::cos(phi) * r, 6, glm::sin(phi) * r);
        const auto vel = 20.0f * glm::normalize(glm::vec3(0, 15 + PCG::RandFloat(PCG::state) * 10, 0) - pos); // Shoot at origin
        SpawnRegularFrog(pos, vel * config.velocityFactor);
      }
    }

    for (auto [entity, transform] : Game::registry.view<Game::ECS::Frog, Game::ECS::Transform>().each())
    {
      if (transform.position.y < -100)
      {
        Game::registry.destroy(entity);
      }
    }

    for (auto [entity, transform] : Game::registry.view<Game::ECS::Dung, Game::ECS::Transform>().each())
    {
      if (transform.position.y < -100)
      {
        Game::registry.destroy(entity);
      }
    }

    for (auto [entity] : Game::registry.view<Game::ECS::PhysicsKinematicAdded>().each())
    {
      const auto& body = Game::registry.get<Game::ECS::PhysicsKinematic>(entity).body;
      Physics::body_interface->AddBody(body, JPH::EActivation::DontActivate);
      Physics::bodyToEntity.emplace(body, entity);
      Game::registry.remove<Game::ECS::PhysicsKinematicAdded>(entity);
    }
    
    for (auto [entity] : Game::registry.view<Game::ECS::PhysicsDynamicAdded>().each())
    {
      // HACK: don't add body to system here since we want to set linear velocity upon creation
      const auto& body = Game::registry.get<Game::ECS::PhysicsDynamic>(entity).body;
      Physics::bodyToEntity.emplace(body, entity);
      Game::registry.remove<Game::ECS::PhysicsDynamicAdded>(entity);
    }
    
    Physics::engine.Update(float(dt), 1, Physics::tempAllocator, Physics::jobSystem);
    Physics::character->PostSimulation(.01f);

    auto bodies = JPH::BodyIDVector();
    //Physics::engine.GetActiveBodies(JPH::EBodyType::RigidBody, bodies);
    Physics::engine.GetBodies(bodies);

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

  void TickGameVariable([[maybe_unused]] double dt, float interpolant)
  {
    // CHEAT: unlock next difficulty
    if (ImGui::GetKeyPressedAmount(ImGuiKey_F5, 100000, 1))
    {
      Game::maxDifficulty++;
    }

    // CHEAT: give player a bunch of money
    if (ImGui::GetKeyPressedAmount(ImGuiKey_F6, 100000, 1))
    {
      Game::moneyCollected += 100;
    }

    bool justUnpaused = false;
    if (Game::state == Game::State::PAUSE && ImGui::GetKeyPressedAmount(ImGuiKey_Escape, 100000, 1))
    {
      Game::SetState(Game::State::GAME);
      justUnpaused = true;
    }

    if (Game::state != Game::State::GAME)
    {
      return;
    }

    Game::secondsSurvived += dt;

    // Unlock new difficulty
    if (Game::secondsSurvived > 30 && Game::difficulty == Game::maxDifficulty)
    {
      Game::maxDifficulty++;
    }

    if (!justUnpaused && ImGui::GetKeyPressedAmount(ImGuiKey_Escape, 100000, 1))
    {
      Game::SetState(Game::State::PAUSE);
    }

    static float lastInterpolant = interpolant;
    const float dtf = static_cast<float>(dt);
    const glm::vec3 camForward = Game::mainCamera.GetForwardDir();
    const auto forward = glm::normalize(glm::vec3{ camForward.x, 0, camForward.z });
    const glm::vec3 right = glm::normalize(glm::cross(forward, { 0, 1, 0 }));
    auto impulse = glm::vec3{};
    if (glfwGetKey(App::window, GLFW_KEY_W) == GLFW_PRESS)
      impulse += forward * Game::playerAcceleration * dtf;
    if (glfwGetKey(App::window, GLFW_KEY_S) == GLFW_PRESS)
      impulse -= forward * Game::playerAcceleration * dtf;
    if (glfwGetKey(App::window, GLFW_KEY_D) == GLFW_PRESS)
      impulse += right * Game::playerAcceleration * dtf;
    if (glfwGetKey(App::window, GLFW_KEY_A) == GLFW_PRESS)
      impulse -= right * Game::playerAcceleration * dtf;

    // Clamp impulse so diagonal input doesn't increase acceleration
    if (auto impulseLen = glm::length(impulse); impulseLen > Game::playerAcceleration)
    {
      impulse = impulse / impulseLen * Game::playerAcceleration;
    }
    auto vel  = Physics::character->GetLinearVelocity();
    // Additionally clamp input velocity so it doesn't increase the horizontal speed beyond the max (instead of clamping final velocity)
    //auto velg = glm::vec3(vel.GetX(), vel.GetY(), vel.GetZ());
    //auto avail = glm::clamp(Game::playerMaxSpeed - glm::length(velg + impulse), 0.0f, 1.0f);
    //impulse *= avail;
    Physics::character->AddLinearVelocity(JPH::Vec3Arg(impulse.x, impulse.y, impulse.z));

    if (Physics::character->GetGroundState() == JPH::CharacterBase::EGroundState::OnGround)
    {
      Game::playerTimeSinceOnGround = 0;
    }
    else
    {
      Game::playerTimeSinceOnGround += dtf;
    }

    // Player tried to jump
    if (Game::playerTimeSinceOnGround <= Game::playerJumpModulationDuration && glfwGetKey(App::window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
      if (Physics::character->GetGroundState() == JPH::CharacterBase::EGroundState::OnGround)
      {
        //auto vel = Physics::character->GetLinearVelocity();
        vel.SetY(Game::playerInitialJumpVelocity); // Remove possible downwards velocity
        Physics::character->SetLinearVelocity(vel);
        auto pos = Physics::character->GetPosition();
        pos.SetY(pos.GetY() + 0.01f); // Unstick the player from the ground
        Physics::character->SetPosition(pos);
        // TODO: play sound
      }

      Physics::character->AddLinearVelocity(JPH::Vec3Arg(0, Game::playerJumpAcceleration * dtf, 0));
    }
    Game::mainCamera.yaw += static_cast<float>(App::cursorOffset.x * Game::cursorSensitivity / 100.0f);
    Game::mainCamera.pitch += static_cast<float>(App::cursorOffset.y * Game::cursorSensitivity / 100.0f);
    Game::mainCamera.pitch = glm::clamp(Game::mainCamera.pitch, -glm::half_pi<float>() + 1e-4f, glm::half_pi<float>() - 1e-4f);

    static glm::vec3 oldCameraPos = Game::mainCamera.position;
    auto p = Physics::character->GetPosition();
    glm::vec3 currentCameraPos = {p.GetX(), p.GetY(), p.GetZ()};

    Game::mainCamera.position = glm::mix(oldCameraPos, currentCameraPos, interpolant);

    if (interpolant < lastInterpolant)
    {
      oldCameraPos = currentCameraPos;
    }

    // Clamp horizontal speed
    if (auto speed = glm::length(glm::vec2(vel.GetX(), vel.GetZ())); speed > Game::playerMaxSpeed)
    {
      vel.SetX(vel.GetX() / speed * Game::playerMaxSpeed);
      vel.SetZ(vel.GetZ() / speed * Game::playerMaxSpeed);
      Physics::character->SetLinearVelocity(vel);
    }

    lastInterpolant = interpolant;
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
    
    dt = glm::min(dt, GAME_TICK);

    App::cursorOffset = {};
    glfwPollEvents();

    gameTickAccum += dt;
    if (gameTickAccum >= GAME_TICK)
    {
      gameTickAccum -= GAME_TICK;

      TickGameFixed(GAME_TICK);
    }

    const float interpolant = glm::min(1.0f, float(gameTickAccum / GAME_TICK));
    TickGameVariable(dt, interpolant);
    TickUI(dt);
    TickRender(dt, interpolant);
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