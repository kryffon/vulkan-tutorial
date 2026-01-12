package main

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math/bits"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:time"
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

validationLayers := [?]cstring{"VK_LAYER_KHRONOS_validation"}
deviceExtensions := [?]cstring{vk.KHR_SWAPCHAIN_EXTENSION_NAME}

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
	window:               glfw.WindowHandle,
	instance:             vk.Instance,
	debugMessenger:       vk.DebugUtilsMessengerEXT,
	surface:              vk.SurfaceKHR,
	physicalDevce:        vk.PhysicalDevice,
	device:               vk.Device,
	graphicsQueue:        vk.Queue,
	presentQueue:         vk.Queue,
	swapChain:            vk.SwapchainKHR,
	swapChainImages:      []vk.Image,
	swapChainImageFormat: vk.Format,
	swapChainExtent:      vk.Extent2D,
	swapChainImageViews:  []vk.ImageView,
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
	createSwapChain(app)
	createImageViews(app)
}

mainLoop :: proc(using app: ^HelloTriangleApplication) {
	start := time.tick_now()
	for !glfw.WindowShouldClose(window) {
		glfw.PollEvents()
		if time.tick_diff(start, time.tick_now()) > 2 * time.Second {
			glfw.SetWindowShouldClose(window, true)
		}
	}
}

cleanup :: proc(using app: ^HelloTriangleApplication) {
	for imageView in swapChainImageViews {
		vk.DestroyImageView(device, imageView, nil)
	}
	delete(swapChainImageViews)

	vk.DestroySwapchainKHR(device, swapChain, nil)
	vk.DestroyDevice(device, nil)
	when ENABLE_VALIDATION_LAYERS {
		vk.DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nil)
	}
	vk.DestroySurfaceKHR(instance, surface, nil)
	vk.DestroyInstance(instance, nil)
	glfw.DestroyWindow(window)
	glfw.Terminate()

	delete(swapChainImages)
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
		sType                   = .DEVICE_CREATE_INFO,
		pQueueCreateInfos       = raw_data(queueCreateInfos),
		queueCreateInfoCount    = u32(len(queueCreateInfos)),
		pEnabledFeatures        = &deviceFeatures,
		enabledExtensionCount   = u32(len(deviceExtensions)),
		ppEnabledExtensionNames = raw_data(deviceExtensions[:]),
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

SwapChainSupportDetails :: struct {
	capabilities: vk.SurfaceCapabilitiesKHR,
	formats:      []vk.SurfaceFormatKHR,
	presentModes: []vk.PresentModeKHR,
}

createSwapChain :: proc(using app: ^HelloTriangleApplication) {
	swapChainSupport: SwapChainSupportDetails = querySwapChainSupport(app, physicalDevce)
	defer delete(swapChainSupport.formats)
	defer delete(swapChainSupport.presentModes)

	surfaceFormat: vk.SurfaceFormatKHR = chooseSwapSurfaceFormat(swapChainSupport.formats)
	presentMode: vk.PresentModeKHR = chooseSwapPresentMode(swapChainSupport.presentModes)
	extent: vk.Extent2D = chooseSwapExtent(app, &swapChainSupport.capabilities)

	imageCount: u32 = swapChainSupport.capabilities.minImageCount + 1
	// maxImageCount == 0 means no upper limit
	if swapChainSupport.capabilities.maxImageCount > 0 {
		imageCount = min(imageCount, swapChainSupport.capabilities.maxImageCount)
	}

	createInfo := vk.SwapchainCreateInfoKHR {
		sType            = .SWAPCHAIN_CREATE_INFO_KHR,
		surface          = surface,
		minImageCount    = imageCount,
		imageFormat      = surfaceFormat.format,
		imageColorSpace  = surfaceFormat.colorSpace,
		imageExtent      = extent,
		// imageArrayLayers > 1 if 3D stereo
		imageArrayLayers = 1,
		imageUsage       = {.COLOR_ATTACHMENT},
	}

	indices := findQueueFamilies(app, physicalDevce)
	queueFamilyIndices := [?]u32{indices.graphicsFamily.?, indices.presentFamily.?}

	if queueFamilyIndices[0] != queueFamilyIndices[1] {
		createInfo.imageSharingMode = .CONCURRENT
		createInfo.queueFamilyIndexCount = 2
		createInfo.pQueueFamilyIndices = raw_data(queueFamilyIndices[:])
	} else {
		createInfo.imageSharingMode = .EXCLUSIVE
	}

	// if we want to add some type of transform(like flipping, etc) before presenting, do here
	createInfo.preTransform = swapChainSupport.capabilities.currentTransform
	createInfo.compositeAlpha = {.OPAQUE}
	createInfo.presentMode = presentMode
	createInfo.clipped = true
	// createInfo.oldSwapchain = 0

	must(vk.CreateSwapchainKHR(device, &createInfo, nil, &swapChain))

	must(vk.GetSwapchainImagesKHR(device, swapChain, &imageCount, nil))
	swapChainImages = make([]vk.Image, imageCount)
	must(vk.GetSwapchainImagesKHR(device, swapChain, &imageCount, raw_data(swapChainImages)))

	swapChainImageFormat = surfaceFormat.format
	swapChainExtent = extent
}

chooseSwapSurfaceFormat :: proc(availableFormats: []vk.SurfaceFormatKHR) -> vk.SurfaceFormatKHR {
	for availableFormat in availableFormats {
		if availableFormat.format == .B8G8R8A8_SRGB &&
		   availableFormat.colorSpace == .SRGB_NONLINEAR {
			return availableFormat
		}
	}
	return availableFormats[0]
}

chooseSwapPresentMode :: proc(availablePresentModes: []vk.PresentModeKHR) -> vk.PresentModeKHR {
	for availablePresentMode in availablePresentModes {
		if availablePresentMode == .MAILBOX {
			return availablePresentMode
		}
	}
	return .FIFO
}

chooseSwapExtent :: proc(
	app: ^HelloTriangleApplication,
	capabilities: ^vk.SurfaceCapabilitiesKHR,
) -> vk.Extent2D {
	if capabilities.currentExtent.width != bits.U32_MAX {
		return capabilities.currentExtent
	} else {
		width, height := glfw.GetFramebufferSize(app.window)
		actualExtent := vk.Extent2D{u32(width), u32(height)}
		actualExtent.width = clamp(
			actualExtent.width,
			capabilities.minImageExtent.width,
			capabilities.maxImageExtent.width,
		)
		actualExtent.height = clamp(
			actualExtent.height,
			capabilities.minImageExtent.height,
			capabilities.maxImageExtent.height,
		)
		return actualExtent
	}
}

querySwapChainSupport :: proc(
	app: ^HelloTriangleApplication,
	device: vk.PhysicalDevice,
) -> SwapChainSupportDetails {
	details: SwapChainSupportDetails

	must(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(device, app.surface, &details.capabilities))

	formatCount: u32
	must(vk.GetPhysicalDeviceSurfaceFormatsKHR(device, app.surface, &formatCount, nil))
	if formatCount != 0 {
		details.formats = make([]vk.SurfaceFormatKHR, formatCount)
		must(
			vk.GetPhysicalDeviceSurfaceFormatsKHR(
				device,
				app.surface,
				&formatCount,
				raw_data(details.formats),
			),
		)
	}

	presentModeCount: u32
	must(vk.GetPhysicalDeviceSurfacePresentModesKHR(device, app.surface, &presentModeCount, nil))
	if presentModeCount != 0 {
		details.presentModes = make([]vk.PresentModeKHR, presentModeCount)
		must(
			vk.GetPhysicalDeviceSurfacePresentModesKHR(
				device,
				app.surface,
				&presentModeCount,
				raw_data(details.presentModes),
			),
		)
	}
	return details
}

createImageViews :: proc(using app: ^HelloTriangleApplication) {
	swapChainImageViews = make([]vk.ImageView, len(swapChainImages))
	for _, i in swapChainImages {
		createInfo := vk.ImageViewCreateInfo {
			sType = .IMAGE_VIEW_CREATE_INFO,
			image = swapChainImages[i],
			viewType = .D2,
			format = swapChainImageFormat,
			components = vk.ComponentMapping {
				r = .IDENTITY,
				g = .IDENTITY,
				b = .IDENTITY,
				a = .IDENTITY,
			},
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		}
		must(vk.CreateImageView(device, &createInfo, nil, &swapChainImageViews[i]))
	}
}

