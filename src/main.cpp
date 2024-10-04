#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <Fwog/Context.h>
#include <Fwog/DebugMarker.h>
#include <Fwog/Shader.h>
#include <Fwog/Pipeline.h>
#include <Fwog/Buffer.h>

#include <miniaudio.h>

#include <filesystem>
#include <optional>
#include <iostream>
#include <fstream>
#include <exception>

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
      glm::vec3 texcoord{};
    };

    struct Mesh
    {
      std::optional<Fwog::TypedBuffer<Vertex>> vertexBuffer;
      std::optional<Fwog::TypedBuffer<index_t>> indexBuffer;
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

layout(binding = 0, std430) readonly buffer VertexBuffer
{
  Vertex vertices[];
};

layout(binding = 1, std430) readonly buffer FrameUniforms
{
  mat4 clip_from_world;
}frame;

struct InstanceUniforms
{
  mat4 world_from_object;
};

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
}
)";
    }
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
    }

    struct View
    {
      glm::vec3 position{};
      float pitch{}; // pitch angle in radians
      float yaw{};   // yaw angle in radians

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
      throw std::exception("Failed to initialize GLFW");
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
    App::window = glfwCreateWindow(static_cast<int>(videoMode->width * .75), static_cast<int>(videoMode->height * .75), "LD56", nullptr, nullptr);
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
  void InitializeRenderer()
  {
    auto vertexShader = Fwog::Shader(Fwog::PipelineStage::VERTEX_SHADER, Render::sceneVertexSource);
    auto fragmentShader = Fwog::Shader(Fwog::PipelineStage::FRAGMENT_SHADER, Render::sceneFragmentSource);
    Render::pipeline = Fwog::GraphicsPipeline({
      .vertexShader = &vertexShader,
      .fragmentShader = &fragmentShader,
      .depthState = {.depthTestEnable = true, .depthWriteEnable = true,},
      });
  }

  void TerminateRenderer()
  {
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
      },
      [&]
      {
        Fwog::Cmd::BindGraphicsPipeline(Render::pipeline.value());
        // Iterate game objects with mesh and transform, then render them
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

    transform.position = { 0, 0, 0 };

    Game::registry.emplace<Game::ECS::MeshRef>(Game::testEntity).mesh = &Game::testMesh;

    Render::Vertex vertices[] = {
      Render::Vertex{{0, 0, 0}},
      Render::Vertex{{1, -1, 0}},
      Render::Vertex{{1, 1, 0}},
    };
    uint32_t indices[] = { 0, 1, 2 };

    Game::testMesh.vertexBuffer.emplace(std::size(vertices));
    Game::testMesh.indexBuffer.emplace(std::size(indices));
  }

  void TerminateGame()
  {

  }

  void TickGameFixed([[maybe_unused]] double dt)
  {

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

    glfwPollEvents();

    gameTickAccum += dt;
    if (gameTickAccum >= GAME_TICK)
    {
      gameTickAccum -= GAME_TICK;

      TickGameFixed(dt);
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
  InitializeGame();
  MainLoop();
  TerminateGame();
  TerminateRenderer();
  TerminateApplication();
}