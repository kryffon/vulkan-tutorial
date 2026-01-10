# NOTES

## steps

1. **setup vkInstance** and **physical device** selection
2. create **logical device** that describes which features should be used i.e. queues, etc
3. create **windows**(glfw), and **swapchains**
    - swapchains is a collection of render targets
    - currently drawn image on screen, and currently being rendered image target are different and are provided to us by the swapchains
    - its a surface that represents the window when i want to draw
4. wrap the swapchain render target into a vkImageView(describes part of the render target that needs to be used). And frame buffer references image views that are to be used for color, depth, etc. There can be many images in swap chains
5. **render passes**: explain the process to the gpu, what to take, what to do
6. **graphics pipline**: configurable state of the gpu, config? shaders, etc
7. **comand pool** and **command buffer**: record the commands(line begin_render_pass, bind_graphics_pipeline, draw, end_render_pass), etc. it is a queue
8. **main loop**
  - get image from swapchain
  - select appropriate command buffer and submit
  - return the image to swapchain for presentation to the screen

## NOTE
1. command can run asynchronously so we need to use semaphores to wait for synchronize the order of execution ourselves
2. For example, object creation generally follows this pattern:
  ```c
  VkXXXCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_XXX_CREATE_INFO;
  createInfo.pNext = nullptr; // extension struction ptr
  createInfo.foo = ...;
  createInfo.bar = ...;
  // allocation callbacks are present for custom allocation

  VkXXX object;
  if (vkCreateXXX(&createInfo, nullptr, &object) != VK_SUCCESS) {
    std::cerr << "failed to create object" << std::endl;
    return false;
  }
  // most functions return VK_SUCCESS or an error code
  ```
3. **validatoin layers** insert pieces of codes between APIs to check function parameters, memory management, etc. Used for debugging
