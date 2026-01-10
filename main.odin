package main

import "base:runtime"
import "core:log"
import "core:mem"
import "core:os"
import "vendor:glfw"
import vk "vendor:vulkan"

WIDTH :: 800
HEIGHT :: 600

HelloTriangleApplication :: struct {
	window:   glfw.WindowHandle,
	instance: vk.Instance,
}

initWindow :: proc(using app: ^HelloTriangleApplication) {
	glfw.SetErrorCallback(glfw_error_callback)
	assert(glfw.Init() == true)
	// donot create opengl context
	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	glfw.WindowHint(glfw.RESIZABLE, glfw.FALSE)
	window = glfw.CreateWindow(WIDTH, HEIGHT, "Vulkan", nil, nil)
}

initVulkan :: proc(using app: ^HelloTriangleApplication) {
	vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))
	assert(vk.CreateInstance != nil, "vulkan function pointers not loaded")

	printVkExtensions(app)

	createInstance(app)
	vk.load_proc_addresses_instance(instance)
}

printVkExtensions :: proc(using app: ^HelloTriangleApplication) {
	extensionCount: u32
	vk.EnumerateInstanceExtensionProperties(nil, &extensionCount, nil)
	// extensions := make([]vk.ExtensionProperties, extensionCount)
	// defer delete(extensions)
	// vk.EnumerateInstanceExtensionProperties(nil, &extensionCount, raw_data(extensions))
	log.infof("VULKAN: %v extensions supported", extensionCount)
	// for e in extensions {
	// 	log.infof("\t%s (version: %v)", e.extensionName, e.specVersion)
	// }
}

createInstance :: proc(using app: ^HelloTriangleApplication) {
	appInfo := vk.ApplicationInfo {
		sType              = .APPLICATION_INFO,
		pApplicationName   = "Hello Triangle",
		applicationVersion = vk.MAKE_VERSION(1, 0, 0),
		pEngineName        = "No Engine",
		engineVersion      = vk.MAKE_VERSION(1, 0, 0),
		apiVersion         = vk.API_VERSION_1_0,
	}

	glfwExtensions := glfw.GetRequiredInstanceExtensions()
	log.infof("GLFW: Extensions: %v", glfwExtensions)

	createInfo := vk.InstanceCreateInfo {
		sType                   = .INSTANCE_CREATE_INFO,
		pApplicationInfo        = &appInfo,
		enabledExtensionCount   = u32(len(glfwExtensions)),
		ppEnabledExtensionNames = raw_data(glfwExtensions),
		enabledLayerCount       = 0,
	}

	must(vk.CreateInstance(&createInfo, nil, &instance))
}

mainLoop :: proc(using app: ^HelloTriangleApplication) {
	for !glfw.WindowShouldClose(window) {
		glfw.PollEvents()
		// glfw.SetWindowShouldClose(window, true)
	}
}

cleanup :: proc(using app: ^HelloTriangleApplication) {
	vk.DestroyInstance(instance, nil)
	glfw.DestroyWindow(window)
	glfw.Terminate()

}

run :: proc(using app: ^HelloTriangleApplication) {
	initWindow(app)
	initVulkan(app)
	mainLoop(app)
	cleanup(app)
}

main :: proc() {
	context.logger = log.create_console_logger(opt = log.Options{.Level})
	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				log.errorf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					log.errorf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}
	app: HelloTriangleApplication
	run(&app)
}

must :: proc(res: vk.Result, location := #caller_location) {
	if res != .SUCCESS {
		log.fatalf("[%v] %v", location, res)
		os.exit(int(res))
	}
}

glfw_error_callback :: proc "c" (code: i32, description: cstring) {
	context = runtime.default_context()
	log.errorf("GLFW: %i: %s", code, description)
}

