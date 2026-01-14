package main

import "base:intrinsics"
import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math/bits"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "vendor:glfw"
import vk "vendor:vulkan"

g_context: runtime.Context

must :: proc(res: vk.Result, location := #caller_location) {
	when ODIN_DEBUG {
		if res != .SUCCESS {
			log.fatalf("[%v] %v", location, res)
			os.exit(int(res))
		}
	}
}

glfw_error_callback :: proc "c" (code: i32, description: cstring) {
	context = g_context
	log.errorf("GLFW: %i: %s", code, description)
}

WIDTH :: 800
HEIGHT :: 600

ENABLE_VALIDATION_LAYERS :: #config(ENABLE_VALIDATION_LAYERS, ODIN_DEBUG)

SHADER_BASIC_VERT :: #load("./shaders/bin/basic.vert.spv")
SHADER_BASIC_FRAG :: #load("./shaders/bin/basic.frag.spv")

MAX_FRAMES_IN_FLIGHT :: 2

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
	window:                   glfw.WindowHandle,
	instance:                 vk.Instance,
	debugMessenger:           vk.DebugUtilsMessengerEXT,
	surface:                  vk.SurfaceKHR,
	physicalDevce:            vk.PhysicalDevice,
	device:                   vk.Device,
	graphicsQueue:            vk.Queue,
	presentQueue:             vk.Queue,
	swapChain:                vk.SwapchainKHR,
	swapChainImages:          []vk.Image,
	swapChainImageFormat:     vk.Format,
	swapChainExtent:          vk.Extent2D,
	swapChainImageViews:      []vk.ImageView,
	swapChainFramebuffers:    []vk.Framebuffer,
	renderPass:               vk.RenderPass,
	pipelineLayout:           vk.PipelineLayout,
	graphicsPipeline:         vk.Pipeline,
	commandPool:              vk.CommandPool,
	vertexBuffer:             vk.Buffer,
	vertexBufferMemory:       vk.DeviceMemory,
	indexBuffer:              vk.Buffer,
	indexBufferMemory:        vk.DeviceMemory,
	commandBuffers:           []vk.CommandBuffer,
	imageAvailableSemaphores: []vk.Semaphore,
	renderFinishedSemaphores: []vk.Semaphore,
	inFlightFences:           []vk.Fence,
	imagesInFlight:           []vk.Fence,
	currentFrame:             int,
	framebufferResized:       bool,
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
	window = glfw.CreateWindow(WIDTH, HEIGHT, "Vulkan", nil, nil)
	glfw.SetWindowUserPointer(window, app)
	glfw.SetFramebufferSizeCallback(window, framebufferResizeCallback)
}

framebufferResizeCallback :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
	app := (^HelloTriangleApplication)(glfw.GetWindowUserPointer(window))
	app.framebufferResized = true
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
	createRenderPass(app)
	createGraphicsPipeline(app)
	createFramebuffers(app)
	createCommandPool(app)
	createVertexBuffer(app)
	createIndexBuffer(app)
	createCommandBuffers(app)
	createSyncObjects(app)
}

mainLoop :: proc(using app: ^HelloTriangleApplication) {
	for !glfw.WindowShouldClose(window) {
		glfw.PollEvents()
		if glfw.GetKey(window, glfw.KEY_ESCAPE) == glfw.PRESS {
			glfw.SetWindowShouldClose(window, true)
		}
		drawFrame(app)
	}
	must(vk.DeviceWaitIdle(device))
}

cleanupSwapChain :: proc(using app: ^HelloTriangleApplication) {
	for framebuffer in swapChainFramebuffers {
		vk.DestroyFramebuffer(device, framebuffer, nil)
	}
	vk.FreeCommandBuffers(device, commandPool, u32(len(commandBuffers)), raw_data(commandBuffers))
	vk.DestroyPipeline(device, graphicsPipeline, nil)
	vk.DestroyPipelineLayout(device, pipelineLayout, nil)
	vk.DestroyRenderPass(device, renderPass, nil)
	for imageView in swapChainImageViews {
		vk.DestroyImageView(device, imageView, nil)
	}
	vk.DestroySwapchainKHR(device, swapChain, nil)
}

cleanup :: proc(using app: ^HelloTriangleApplication) {
	cleanupSwapChain(app)
	vk.DestroyBuffer(device, indexBuffer, nil)
	vk.FreeMemory(device, indexBufferMemory, nil)
	vk.DestroyBuffer(device, vertexBuffer, nil)
	vk.FreeMemory(device, vertexBufferMemory, nil)
	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		vk.DestroySemaphore(device, renderFinishedSemaphores[i], nil)
		vk.DestroySemaphore(device, imageAvailableSemaphores[i], nil)
		vk.DestroyFence(device, inFlightFences[i], nil)
	}

	vk.DestroyCommandPool(device, commandPool, nil)
	vk.DestroyDevice(device, nil)
	when ENABLE_VALIDATION_LAYERS {
		vk.DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nil)
	}
	vk.DestroySurfaceKHR(instance, surface, nil)
	vk.DestroyInstance(instance, nil)
	glfw.DestroyWindow(window)
	glfw.Terminate()

	delete(commandBuffers)
	delete(swapChainFramebuffers)
	delete(swapChainImages)
	delete(swapChainImageViews)
	delete(renderFinishedSemaphores)
	delete(imageAvailableSemaphores)
	delete(inFlightFences)
	delete(imagesInFlight)
}

recreateSwapChain :: proc(using app: ^HelloTriangleApplication) {
	width, height := glfw.GetFramebufferSize(window)
	for width == 0 || height == 0 {
		width, height = glfw.GetFramebufferSize(window)
		glfw.WaitEventsTimeout(0)
	}
	vk.DeviceWaitIdle(device)
	cleanupSwapChain(app)

	createSwapChain(app)
	createImageViews(app)
	createRenderPass(app)
	createGraphicsPipeline(app)
	createFramebuffers(app)
	createCommandBuffers(app)

	// reserve(&imagesInFlight, len(swapChainImages))
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
		if isDeviceSuitable(surface, device) {
			physicalDevce = device
			break
		}
	}
	log.assertf(physicalDevce != nil, "failed to find a suitable GPU!")
}

isDeviceSuitable :: proc(surface: vk.SurfaceKHR, device: vk.PhysicalDevice) -> bool {
	// props: vk.PhysicalDeviceProperties
	// vk.GetPhysicalDeviceProperties(device, &props)
	// log.debugf("VULKAN: device: %s type:%v", props.deviceName, props.deviceType)
	indices := findQueueFamilies(surface, device)
	return isComplete(indices)
}

findQueueFamilies :: proc(
	surface: vk.SurfaceKHR,
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
		must(vk.GetPhysicalDeviceSurfaceSupportKHR(device, u32(i), surface, &presentSupport))
		if presentSupport {
			indices.presentFamily = u32(i)
		}
		if isComplete(indices) do break
	}
	return indices
}

createLogicalDevice :: proc(using app: ^HelloTriangleApplication) {
	indices := findQueueFamilies(surface, physicalDevce)
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
	swapChainSupport: SwapChainSupportDetails = querySwapChainSupport(surface, physicalDevce)
	defer delete(swapChainSupport.formats)
	defer delete(swapChainSupport.presentModes)

	surfaceFormat: vk.SurfaceFormatKHR = chooseSwapSurfaceFormat(swapChainSupport.formats)
	presentMode: vk.PresentModeKHR = chooseSwapPresentMode(swapChainSupport.presentModes)
	extent: vk.Extent2D = chooseSwapExtent(window, &swapChainSupport.capabilities)

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

	indices := findQueueFamilies(surface, physicalDevce)
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
	window: glfw.WindowHandle,
	capabilities: ^vk.SurfaceCapabilitiesKHR,
) -> vk.Extent2D {
	if capabilities.currentExtent.width != bits.U32_MAX {
		return capabilities.currentExtent
	} else {
		width, height := glfw.GetFramebufferSize(window)
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
	surface: vk.SurfaceKHR,
	device: vk.PhysicalDevice,
) -> SwapChainSupportDetails {
	details: SwapChainSupportDetails

	must(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities))

	formatCount: u32
	must(vk.GetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nil))
	if formatCount != 0 {
		details.formats = make([]vk.SurfaceFormatKHR, formatCount)
		must(
			vk.GetPhysicalDeviceSurfaceFormatsKHR(
				device,
				surface,
				&formatCount,
				raw_data(details.formats),
			),
		)
	}

	presentModeCount: u32
	must(vk.GetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nil))
	if presentModeCount != 0 {
		details.presentModes = make([]vk.PresentModeKHR, presentModeCount)
		must(
			vk.GetPhysicalDeviceSurfacePresentModesKHR(
				device,
				surface,
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

createGraphicsPipeline :: proc(using app: ^HelloTriangleApplication) {
	vertShaderModule: vk.ShaderModule = createShaderModule(device, SHADER_BASIC_VERT)
	fragShaderModule: vk.ShaderModule = createShaderModule(device, SHADER_BASIC_FRAG)
	defer {
		vk.DestroyShaderModule(device, fragShaderModule, nil)
		vk.DestroyShaderModule(device, vertShaderModule, nil)
	}

	vertShaderStageCreateInfo := vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.VERTEX},
		module = vertShaderModule,
		pName  = "main",
	}
	fragShaderStageCreateInfo := vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.FRAGMENT},
		module = fragShaderModule,
		pName  = "main",
	}

	shaderStages := [?]vk.PipelineShaderStageCreateInfo {
		vertShaderStageCreateInfo,
		fragShaderStageCreateInfo,
	}

	bindingDescription := getBindingDescription(Vertex)
	attributeDescription := getAttributeDescription(Vertex)
	defer delete(attributeDescription)

	vertexInputInfo := vk.PipelineVertexInputStateCreateInfo {
		sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount   = 1,
		pVertexBindingDescriptions      = &bindingDescription,
		vertexAttributeDescriptionCount = u32(len(attributeDescription)),
		pVertexAttributeDescriptions    = raw_data(attributeDescription),
	}

	inputAssembly := vk.PipelineInputAssemblyStateCreateInfo {
		sType                  = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology               = .TRIANGLE_LIST,
		primitiveRestartEnable = false,
	}

	viewport := vk.Viewport {
		x        = 0,
		y        = 0,
		width    = f32(swapChainExtent.width),
		height   = f32(swapChainExtent.height),
		minDepth = 0,
		maxDepth = 1,
	}

	scissor := vk.Rect2D {
		offset = {0, 0},
		extent = swapChainExtent,
	}

	viewportState := vk.PipelineViewportStateCreateInfo {
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		pViewports    = &viewport,
		scissorCount  = 1,
		pScissors     = &scissor,
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo {
		sType                   = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		depthClampEnable        = false,
		rasterizerDiscardEnable = false,
		polygonMode             = .FILL,
		lineWidth               = 1.0,
		cullMode                = {.BACK},
		frontFace               = .CLOCKWISE,
		depthBiasEnable         = false,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo {
		sType                = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		sampleShadingEnable  = false,
		rasterizationSamples = {._1},
	}

	colorBlendAttachment := vk.PipelineColorBlendAttachmentState {
		colorWriteMask = {.R, .G, .B, .A},
		blendEnable    = false,
	}

	colorBlending := vk.PipelineColorBlendStateCreateInfo {
		sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		logicOpEnable   = false,
		logicOp         = .COPY,
		attachmentCount = 1,
		pAttachments    = &colorBlendAttachment,
		blendConstants  = {0, 0, 0, 0},
	}

	pipelineLayoutInfo := vk.PipelineLayoutCreateInfo {
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount         = 0,
		pushConstantRangeCount = 0,
	}

	must(vk.CreatePipelineLayout(device, &pipelineLayoutInfo, nil, &pipelineLayout))

	pipelineInfo := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		stageCount          = 2,
		pStages             = raw_data(shaderStages[:]),
		pVertexInputState   = &vertexInputInfo,
		pInputAssemblyState = &inputAssembly,
		pViewportState      = &viewportState,
		pRasterizationState = &rasterizer,
		pMultisampleState   = &multisampling,
		pColorBlendState    = &colorBlending,
		layout              = pipelineLayout,
		renderPass          = renderPass,
		subpass             = 0,
		// basePipelineHandle = 0,
		basePipelineIndex   = -1,
	}

	must(vk.CreateGraphicsPipelines(device, 0, 1, &pipelineInfo, nil, &graphicsPipeline))
}

createShaderModule :: proc(device: vk.Device, code: []u8) -> vk.ShaderModule {
	createInfo := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(code),
		pCode    = raw_data(slice.reinterpret([]u32, code)),
	}

	shaderModule: vk.ShaderModule
	must(vk.CreateShaderModule(device, &createInfo, nil, &shaderModule))
	return shaderModule
}

createRenderPass :: proc(using app: ^HelloTriangleApplication) {
	colorAttachment := vk.AttachmentDescription {
		format         = swapChainImageFormat,
		samples        = {._1},
		loadOp         = .CLEAR,
		storeOp        = .STORE,
		stencilLoadOp  = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout  = .UNDEFINED,
		finalLayout    = .PRESENT_SRC_KHR,
	}

	colorAttachmentRef := vk.AttachmentReference {
		attachment = 0,
		layout     = .COLOR_ATTACHMENT_OPTIMAL,
	}

	subpass := vk.SubpassDescription {
		pipelineBindPoint    = .GRAPHICS,
		colorAttachmentCount = 1,
		pColorAttachments    = &colorAttachmentRef,
	}

	dependency := vk.SubpassDependency {
		srcSubpass    = vk.SUBPASS_EXTERNAL,
		dstSubpass    = 0,
		srcStageMask  = {.COLOR_ATTACHMENT_OUTPUT},
		// srcAccessMask = {.INDIRECT_COMMAND_READ},
		dstStageMask  = {.COLOR_ATTACHMENT_OUTPUT},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
	}

	renderPassInfo := vk.RenderPassCreateInfo {
		sType           = .RENDER_PASS_CREATE_INFO,
		attachmentCount = 1,
		pAttachments    = &colorAttachment,
		subpassCount    = 1,
		pSubpasses      = &subpass,
		dependencyCount = 1,
		pDependencies   = &dependency,
	}

	must(vk.CreateRenderPass(device, &renderPassInfo, nil, &renderPass))
}

createFramebuffers :: proc(using app: ^HelloTriangleApplication) {
	swapChainFramebuffers = make([]vk.Framebuffer, len(swapChainImageViews))

	for _, i in swapChainImageViews {
		attachments := [?]vk.ImageView{swapChainImageViews[i]}

		framebufferInfo := vk.FramebufferCreateInfo {
			sType           = .FRAMEBUFFER_CREATE_INFO,
			renderPass      = renderPass,
			attachmentCount = 1,
			pAttachments    = raw_data(attachments[:]),
			width           = swapChainExtent.width,
			height          = swapChainExtent.height,
			layers          = 1,
		}
		must(vk.CreateFramebuffer(device, &framebufferInfo, nil, &swapChainFramebuffers[i]))
	}
}

createCommandPool :: proc(using app: ^HelloTriangleApplication) {
	queueFamilyIndices := findQueueFamilies(surface, physicalDevce)
	poolInfo := vk.CommandPoolCreateInfo {
		sType            = .COMMAND_POOL_CREATE_INFO,
		queueFamilyIndex = queueFamilyIndices.graphicsFamily.?,
	}
	must(vk.CreateCommandPool(device, &poolInfo, nil, &commandPool))
}

createCommandBuffers :: proc(using app: ^HelloTriangleApplication) {
	commandBuffers = make([]vk.CommandBuffer, len(swapChainFramebuffers))

	allocInfo := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = commandPool,
		level              = .PRIMARY,
		commandBufferCount = u32(len(commandBuffers)),
	}
	must(vk.AllocateCommandBuffers(device, &allocInfo, raw_data(commandBuffers)))

	for _, i in commandBuffers {
		beginInfo := vk.CommandBufferBeginInfo {
			sType = .COMMAND_BUFFER_BEGIN_INFO,
		}

		must(vk.BeginCommandBuffer(commandBuffers[i], &beginInfo))

		clearColor := vk.ClearValue {
			color = vk.ClearColorValue{float32 = [4]f32{0, 0, 0, 1}},
		}
		renderPassInfo := vk.RenderPassBeginInfo {
			sType = .RENDER_PASS_BEGIN_INFO,
			renderPass = renderPass,
			framebuffer = swapChainFramebuffers[i],
			renderArea = {offset = {0, 0}, extent = swapChainExtent},
			clearValueCount = 1,
			pClearValues = &clearColor,
		}

		vk.CmdBeginRenderPass(commandBuffers[i], &renderPassInfo, .INLINE)
		{
			vk.CmdBindPipeline(commandBuffers[i], .GRAPHICS, graphicsPipeline)

			vertexBuffers := [?]vk.Buffer{vertexBuffer}
			offsets := [?]vk.DeviceSize{0}
			vk.CmdBindVertexBuffers(
				commandBuffers[i],
				0,
				1,
				raw_data(vertexBuffers[:]),
				raw_data(offsets[:]),
			)

			vk.CmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, .UINT16)
			vk.CmdDrawIndexed(commandBuffers[i], u32(len(indices)), 1, 0, 0, 0)
		}
		vk.CmdEndRenderPass(commandBuffers[i])
		must(vk.EndCommandBuffer(commandBuffers[i]))
	}
}

createSyncObjects :: proc(using app: ^HelloTriangleApplication) {
	imageAvailableSemaphores = make([]vk.Semaphore, MAX_FRAMES_IN_FLIGHT)
	renderFinishedSemaphores = make([]vk.Semaphore, MAX_FRAMES_IN_FLIGHT)
	inFlightFences = make([]vk.Fence, MAX_FRAMES_IN_FLIGHT)
	imagesInFlight = make([]vk.Fence, len(swapChainImages))

	semaphoreInfo := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
	}
	fenceInfo := vk.FenceCreateInfo {
		sType = .FENCE_CREATE_INFO,
		flags = {.SIGNALED},
	}

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		must(vk.CreateSemaphore(device, &semaphoreInfo, nil, &imageAvailableSemaphores[i]))
		must(vk.CreateSemaphore(device, &semaphoreInfo, nil, &renderFinishedSemaphores[i]))
		must(vk.CreateFence(device, &fenceInfo, nil, &inFlightFences[i]))
	}
}

drawFrame :: proc(using app: ^HelloTriangleApplication) {
	must(vk.WaitForFences(device, 1, raw_data(inFlightFences), true, bits.U64_MAX))

	imageIndex: u32
	// must(
	vk.AcquireNextImageKHR(
		device,
		swapChain,
		bits.U64_MAX,
		imageAvailableSemaphores[currentFrame],
		0,
		&imageIndex,
	)
	// )

	if imagesInFlight[imageIndex] != 0 {
		must(vk.WaitForFences(device, 1, &imagesInFlight[imageIndex], true, bits.U64_MAX))
	}
	imagesInFlight[imageIndex] = inFlightFences[currentFrame]

	waitSemaphores := [?]vk.Semaphore{imageAvailableSemaphores[currentFrame]}
	waitStates := [?]vk.PipelineStageFlags{{.COLOR_ATTACHMENT_OUTPUT}}
	signalSemaphores := [?]vk.Semaphore{renderFinishedSemaphores[currentFrame]}
	submitInfo := vk.SubmitInfo {
		sType                = .SUBMIT_INFO,
		waitSemaphoreCount   = 1,
		pWaitSemaphores      = raw_data(waitSemaphores[:]),
		pWaitDstStageMask    = raw_data(waitStates[:]),
		commandBufferCount   = 1,
		pCommandBuffers      = &commandBuffers[imageIndex],
		signalSemaphoreCount = 1,
		pSignalSemaphores    = raw_data(signalSemaphores[:]),
	}


	must(vk.ResetFences(device, 1, &inFlightFences[currentFrame]))
	must(vk.QueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]))

	swapChains := [?]vk.SwapchainKHR{swapChain}
	presentInfo := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = raw_data(signalSemaphores[:]),
		swapchainCount     = 1,
		pSwapchains        = raw_data(swapChains[:]),
		pImageIndices      = &imageIndex,
	}
	vk.QueuePresentKHR(presentQueue, &presentInfo)
	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT
}

Vertex :: struct {
	pos:   [2]f32,
	color: [3]f32,
}

getBindingDescription :: proc($T: typeid) -> vk.VertexInputBindingDescription {
	when T == Vertex {
		bindingDescription := vk.VertexInputBindingDescription {
			binding   = 0,
			stride    = size_of(T),
			inputRate = .VERTEX,
		}
		return bindingDescription
	}
	unimplemented()
}

getAttributeDescription :: proc($T: typeid) -> []vk.VertexInputAttributeDescription {
	when T == Vertex {
		attributeDescriptions := make([]vk.VertexInputAttributeDescription, 2)
		attributeDescriptions[0] = {
			binding  = 0,
			location = 0,
			format   = .R32G32_SFLOAT,
			offset   = u32(offset_of(Vertex, pos)),
		}
		attributeDescriptions[1] = {
			binding  = 0,
			location = 1,
			format   = .R32G32B32_SFLOAT,
			offset   = u32(offset_of(Vertex, color)),
		}
		return attributeDescriptions
	}
	unimplemented()
}

// odinfmt:disable
vertices := [?]Vertex{
	{{-0.5, -0.5}, {1, 0, 0}},
	{{0.5, -0.5}, {0, 1, 0}},
	{{0.5, 0.5}, {0, 0, 1}},
	{{-0.5, 0.5}, {1, 1, 1}},
}

indices := [?]u16{0,1,2,2,3,0}
// odinfmt:enable

createVertexBuffer :: proc(using app: ^HelloTriangleApplication) {
	bufferSize: vk.DeviceSize = size_of(vertices[0]) * len(vertices)

	stagingBuffer: vk.Buffer
	stagingBufferMemory: vk.DeviceMemory
	createBuffer(
		app,
		bufferSize,
		{.TRANSFER_SRC},
		{.HOST_VISIBLE, .HOST_COHERENT},
		&stagingBuffer,
		&stagingBufferMemory,
	)
	{
		data: rawptr
		must(vk.MapMemory(device, stagingBufferMemory, 0, bufferSize, {}, &data))
		intrinsics.mem_copy_non_overlapping(data, raw_data(vertices[:]), bufferSize)
		vk.UnmapMemory(device, stagingBufferMemory)
	}

	createBuffer(
		app,
		bufferSize,
		{.TRANSFER_DST, .VERTEX_BUFFER},
		{.DEVICE_LOCAL},
		&vertexBuffer,
		&vertexBufferMemory,
	)
	copyBuffer(app, stagingBuffer, vertexBuffer, bufferSize)

	vk.DestroyBuffer(device, stagingBuffer, nil)
	vk.FreeMemory(device, stagingBufferMemory, nil)
}

createBuffer :: proc(
	app: ^HelloTriangleApplication,
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	properties: vk.MemoryPropertyFlags,
	buffer: ^vk.Buffer,
	bufferMemory: ^vk.DeviceMemory,
) {
	bufferInfo := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = size_of(vertices[0]) * len(vertices),
		usage       = usage,
		sharingMode = .EXCLUSIVE,
	}
	must(vk.CreateBuffer(app.device, &bufferInfo, nil, buffer))

	memRequirements: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(app.device, buffer^, &memRequirements)

	allocInfo := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = memRequirements.size,
		memoryTypeIndex = findMemoryType(
			app.physicalDevce,
			memRequirements.memoryTypeBits,
			properties,
		),
	}
	must(vk.AllocateMemory(app.device, &allocInfo, nil, bufferMemory))
	must(vk.BindBufferMemory(app.device, buffer^, bufferMemory^, 0))
}

copyBuffer :: proc(
	app: ^HelloTriangleApplication,
	srcBuffer, dstBuffer: vk.Buffer,
	size: vk.DeviceSize,
) {
	allocInfo := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		level              = .PRIMARY,
		commandPool        = app.commandPool,
		commandBufferCount = 1,
	}

	commandBuffer: vk.CommandBuffer
	must(vk.AllocateCommandBuffers(app.device, &allocInfo, &commandBuffer))

	beginInfo := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}

	must(vk.BeginCommandBuffer(commandBuffer, &beginInfo))

	copyRegion := vk.BufferCopy {
		size = size,
	}
	vk.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion)

	must(vk.EndCommandBuffer(commandBuffer))

	submitInfo := vk.SubmitInfo {
		sType              = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers    = &commandBuffer,
	}

	must(vk.QueueSubmit(app.graphicsQueue, 1, &submitInfo, 0))
	must(vk.QueueWaitIdle(app.graphicsQueue))

	vk.FreeCommandBuffers(app.device, app.commandPool, 1, &commandBuffer)
}

findMemoryType :: proc(
	physicalDevice: vk.PhysicalDevice,
	typeFilter: u32,
	properties: vk.MemoryPropertyFlags,
) -> u32 {
	memProperties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties)

	for i in 0 ..< memProperties.memoryTypeCount {
		if (typeFilter & (1 << i)) > 0 {
			if properties <= memProperties.memoryTypes[i].propertyFlags {
				return i
			}
		}
	}
	assert(false, "failed to find suitable memory type!")
	return 0
}

createIndexBuffer :: proc(using app: ^HelloTriangleApplication) {
	bufferSize: vk.DeviceSize = size_of(indices[0]) * len(indices)

	stagingBuffer: vk.Buffer
	stagingBufferMemory: vk.DeviceMemory
	createBuffer(
		app,
		bufferSize,
		{.TRANSFER_SRC},
		{.HOST_VISIBLE, .HOST_COHERENT},
		&stagingBuffer,
		&stagingBufferMemory,
	)
	{
		data: rawptr
		must(vk.MapMemory(device, stagingBufferMemory, 0, bufferSize, {}, &data))
		intrinsics.mem_copy_non_overlapping(data, raw_data(indices[:]), bufferSize)
		vk.UnmapMemory(device, stagingBufferMemory)
	}

	createBuffer(
		app,
		bufferSize,
		{.TRANSFER_DST, .INDEX_BUFFER},
		{.DEVICE_LOCAL},
		&indexBuffer,
		&indexBufferMemory,
	)
	copyBuffer(app, stagingBuffer, indexBuffer, bufferSize)
	vk.DestroyBuffer(device, stagingBuffer, nil)
	vk.FreeMemory(device, stagingBufferMemory, nil)
}

