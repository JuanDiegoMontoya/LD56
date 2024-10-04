#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <Fwog/Context.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>

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

void InitializeRenderer()
{
  
}

void TickGame([[maybe_unused]] double dt)
{
  
}

void TickRender([[maybe_unused]] double dt)
{
  Fwog::RenderToSwapchain(
    Fwog::SwapchainRenderInfo{
      .name = "Render Triangle",
      .viewport = Fwog::Viewport{.drawRect{.offset = {0, 0}, .extent = {(uint32_t)App::windowSize.x, (uint32_t)App::windowSize.y}}},
      .colorLoadOp = Fwog::AttachmentLoadOp::CLEAR,
      .clearColorValue = {.2f, .0f, .2f, 1.0f},
    },
    [&]
    {
    });

  glfwSwapBuffers(App::window);
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

      TickGame(dt);
    }

    TickRender(dt);
  }
}

int main()
{
  InitializeApplication();
  InitializeRenderer();
  MainLoop();

  //gladLoadGL(glfwGetProcAddress);
  //ImGui::Button("asdf");
  //Fwog::Initialize();
  //entt::registry asdf;
  //[[maybe_unused]] auto asdff = asdf.create();
  //glm::vec3{};
}