package main

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "vendor:glfw"
import vk "vendor:vulkan"

g_context: runtime.Context

must :: proc(res: vk.Result, location := #caller_location) {
	if res != .SUCCESS {
		log.fatalf("[%v] %v", location, res)
		os.exit(int(res))
	}
}

glfw_error_callback :: proc "c" (code: i32, description: cstring) {
	context = g_context
	log.errorf("GLFW: %i: %s", code, description)
}

WIDTH :: 800
HEIGHT :: 600

ENABLE_VALIDATION_LAYERS :: #config(ENABLE_VALIDATION_LAYERS, ODIN_DEBUG)

main :: proc() {
	context.logger = log.create_console_logger(opt = log.Options{.Level})
	g_context = context
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

HelloTriangleApplication :: struct {
	window:         glfw.WindowHandle,
	instance:       vk.Instance,
	debugMessenger: vk.DebugUtilsMessengerEXT,
	surface:        vk.SurfaceKHR,
	physicalDevce:  vk.PhysicalDevice,
	device:         vk.Device,
	graphicsQueue:  vk.Queue,
	presentQueue:   vk.Queue,
}

run :: proc(using app: ^HelloTriangleApplication) {
	initWindow(app)
	initVulkan(app)
	mainLoop(app)
	cleanup(app)
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
	// this load_proc is needed
	vk.load_proc_addresses_instance(instance)

	setupDebugMessenger(app)
	createSurface(app)
	pickPhysicalDevice(app)
	createLogicalDevice(app)
}

mainLoop :: proc(using app: ^HelloTriangleApplication) {
	for !glfw.WindowShouldClose(window) {
		glfw.PollEvents()
		glfw.SetWindowShouldClose(window, true)
	}
}

cleanup :: proc(using app: ^HelloTriangleApplication) {
	vk.DestroyDevice(device, nil)
	when ENABLE_VALIDATION_LAYERS {
		vk.DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nil)
	}
	vk.DestroySurfaceKHR(instance, surface, nil)
	vk.DestroyInstance(instance, nil)
	glfw.DestroyWindow(window)
	glfw.Terminate()

}

printVkExtensions :: proc(using app: ^HelloTriangleApplication) {
	extensionCount: u32
	vk.EnumerateInstanceExtensionProperties(nil, &extensionCount, nil)
	log.infof("VULKAN: %v extensions supported", extensionCount)
	// extensions := make([]vk.ExtensionProperties, extensionCount)
	// defer delete(extensions)
	// vk.EnumerateInstanceExtensionProperties(nil, &extensionCount, raw_data(extensions))
	// for e in extensions {
	// 	log.infof("\t%s (version: %v)", e.extensionName, e.specVersion)
	// }
}

validationLayers := [?]cstring{"VK_LAYER_KHRONOS_validation"}

createInstance :: proc(using app: ^HelloTriangleApplication) {
	when ENABLE_VALIDATION_LAYERS {
		assert(checkValidationLayerSupport())
	}

	appInfo := vk.ApplicationInfo {
		sType              = .APPLICATION_INFO,
		pApplicationName   = "Hello Triangle",
		applicationVersion = vk.MAKE_VERSION(1, 0, 0),
		pEngineName        = "No Engine",
		engineVersion      = vk.MAKE_VERSION(1, 0, 0),
		apiVersion         = vk.API_VERSION_1_0,
	}

	extensions := getRequiredExtensions()
	defer delete(extensions)

	createInfo := vk.InstanceCreateInfo {
		sType                   = .INSTANCE_CREATE_INFO,
		pApplicationInfo        = &appInfo,
		enabledExtensionCount   = u32(len(extensions)),
		ppEnabledExtensionNames = raw_data(extensions),
		enabledLayerCount       = 0,
	}
	when ENABLE_VALIDATION_LAYERS {
		createInfo.enabledLayerCount = u32(len(validationLayers))
		createInfo.ppEnabledLayerNames = raw_data(validationLayers[:])

		debugCreateInfo: vk.DebugUtilsMessengerCreateInfoEXT
		populateDebugMessengerCreateInfo(&debugCreateInfo)
		createInfo.pNext = &debugCreateInfo
	}

	must(vk.CreateInstance(&createInfo, nil, &instance))
}

populateDebugMessengerCreateInfo :: proc(createInfo: ^vk.DebugUtilsMessengerCreateInfoEXT) {
	createInfo.sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT
	createInfo.messageSeverity = {.VERBOSE, .INFO, .WARNING, .ERROR}
	createInfo.messageType = {.GENERAL, .VALIDATION, .PERFORMANCE}
	createInfo.pfnUserCallback = debugCallback
}

setupDebugMessenger :: proc(using app: ^HelloTriangleApplication) {
	when ENABLE_VALIDATION_LAYERS {
		createInfo: vk.DebugUtilsMessengerCreateInfoEXT
		populateDebugMessengerCreateInfo(&createInfo)
		must(vk.CreateDebugUtilsMessengerEXT(instance, &createInfo, nil, &debugMessenger))
	}
}

getRequiredExtensions :: proc() -> []cstring {
	extensions := slice.clone_to_dynamic(glfw.GetRequiredInstanceExtensions())
	log.infof("GLFW: %v extensions supported", len(extensions))
	when ENABLE_VALIDATION_LAYERS {
		append(&extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
	}
	return extensions[:]
}

checkValidationLayerSupport :: proc() -> bool {
	layerCount: u32
	must(vk.EnumerateInstanceLayerProperties(&layerCount, nil))
	log.infof("VULKAN: %v validation layers supported", layerCount)
	availableLayers := make([]vk.LayerProperties, layerCount)
	defer delete(availableLayers)
	must(vk.EnumerateInstanceLayerProperties(&layerCount, raw_data(availableLayers)))
	// for l in availableLayers {
	// 	log.infof("VULKAN: available validation layer: %s", l.layerName)
	// }

	for layerName in validationLayers {
		layerFound := false
		n := len(layerName)
		s := make([]u8, n)
		defer delete(s)
		for layerProperties in availableLayers {
			for i in 0 ..< n {
				s[i] = layerProperties.layerName[i]
			}
			if layerName == cstring(raw_data(s)) {
				layerFound = true
				break
			}
		}
		if !layerFound do return false
	}
	return true
}

debugCallback :: proc "system" (
	messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
	messageTypes: vk.DebugUtilsMessageTypeFlagsEXT,
	pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
	pUserData: rawptr,
) -> b32 {
	context = g_context

	level: log.Level
	if .ERROR in messageSeverity {
		level = .Error
	} else if .WARNING in messageSeverity {
		level = .Warning
	} else if .INFO in messageSeverity {
		level = .Info
	} else {
		level = .Debug
	}

	b: strings.Builder
	strings.builder_init_none(&b, context.temp_allocator)
	defer strings.builder_destroy(&b)
	first := false
	for t in messageTypes {
		if first do strings.write_byte(&b, '|')
		strings.write_string(&b, fmt.tprintf("%v", t))
		first = true
	}
	log.logf(level, "VULKAN: [%s]: %s", strings.to_string(b), pCallbackData.pMessage)
	return false
}

QueueFamilyIndices :: struct {
	graphicsFamily: Maybe(u32),
	presentFamily:  Maybe(u32),
}

isComplete :: proc(q: QueueFamilyIndices) -> bool {
	_, has_g := q.graphicsFamily.?
	_, has_p := q.presentFamily.?
	return has_g && has_p
}

pickPhysicalDevice :: proc(using app: ^HelloTriangleApplication) {
	deviceCount: u32
	must(vk.EnumeratePhysicalDevices(instance, &deviceCount, nil))
	log.assertf(deviceCount > 0, "failed to find GPUs with vulkan support")
	log.infof("VULKAN: %d devices found", deviceCount)

	devices := make([]vk.PhysicalDevice, deviceCount)
	defer delete(devices)
	must(vk.EnumeratePhysicalDevices(instance, &deviceCount, raw_data(devices)))

	for device in devices {
		if isDeviceSuitable(app, device) {
			physicalDevce = device
			break
		}
	}
	log.assertf(physicalDevce != nil, "failed to find a suitable GPU!")
}

isDeviceSuitable :: proc(app: ^HelloTriangleApplication, device: vk.PhysicalDevice) -> bool {
	// props: vk.PhysicalDeviceProperties
	// vk.GetPhysicalDeviceProperties(device, &props)
	// log.debugf("VULKAN: device: %s type:%v", props.deviceName, props.deviceType)
	indices := findQueueFamilies(app, device)
	return isComplete(indices)
}

findQueueFamilies :: proc(
	app: ^HelloTriangleApplication,
	device: vk.PhysicalDevice,
) -> QueueFamilyIndices {
	indices: QueueFamilyIndices
	queueFamilyCount: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nil)
	// log.infof("VULKAN: %d families found for device %v", queueFamilyCount, device)

	queueFamilies := make([]vk.QueueFamilyProperties, queueFamilyCount)
	defer delete(queueFamilies)
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, raw_data(queueFamilies))

	// for q in queueFamilies {
	// 	log.debugf("VULKAN: device: %v queueFamily: %v", device, q.queueFlags)
	// }

	for queueFamily, i in queueFamilies {
		if .GRAPHICS in queueFamily.queueFlags {
			indices.graphicsFamily = u32(i)
		}
		presentSupport: b32
		must(vk.GetPhysicalDeviceSurfaceSupportKHR(device, u32(i), app.surface, &presentSupport))
		if presentSupport {
			indices.presentFamily = u32(i)
		}
		if isComplete(indices) do break
	}
	return indices
}

createLogicalDevice :: proc(using app: ^HelloTriangleApplication) {
	indices := findQueueFamilies(app, physicalDevce)
	uniqueQueueFamilies: [dynamic]u32
	defer delete(uniqueQueueFamilies)
	append(&uniqueQueueFamilies, indices.graphicsFamily.?)
	if uniqueQueueFamilies[0] != indices.presentFamily.? {
		append(&uniqueQueueFamilies, indices.presentFamily.?)
	}

	queueCreateInfos := make([]vk.DeviceQueueCreateInfo, len(uniqueQueueFamilies))
	defer delete(queueCreateInfos)

	queuePriority: f32 = 1.0
	for queueFamily, i in uniqueQueueFamilies {
		queueCreateInfos[i] = vk.DeviceQueueCreateInfo {
			sType            = .DEVICE_QUEUE_CREATE_INFO,
			queueFamilyIndex = queueFamily,
			queueCount       = 1,
			pQueuePriorities = &queuePriority,
		}
	}

	deviceFeatures: vk.PhysicalDeviceFeatures

	createInfo := vk.DeviceCreateInfo {
		sType                 = .DEVICE_CREATE_INFO,
		pQueueCreateInfos     = raw_data(queueCreateInfos),
		queueCreateInfoCount  = u32(len(queueCreateInfos)),
		pEnabledFeatures      = &deviceFeatures,
		enabledExtensionCount = 0,
	}

	when ENABLE_VALIDATION_LAYERS {
		createInfo.enabledLayerCount = u32(len(validationLayers))
		createInfo.ppEnabledLayerNames = raw_data(validationLayers[:])
	} else {
		createInfo.enabledLayerCount = 0
	}

	must(vk.CreateDevice(physicalDevce, &createInfo, nil, &device))
	vk.GetDeviceQueue(device, indices.graphicsFamily.?, 0, &graphicsQueue)
	vk.GetDeviceQueue(device, indices.presentFamily.?, 0, &presentQueue)
}

createSurface :: proc(using app: ^HelloTriangleApplication) {
	must(glfw.CreateWindowSurface(instance, window, nil, &surface))
}

