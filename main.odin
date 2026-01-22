package main

import "base:intrinsics"
import "base:runtime"
import "core:fmt"
import "core:image/png"
import "core:log"
import "core:math"
import "core:math/bits"
import "core:math/linalg/glsl"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:time"
import "tinyobj"
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

FPS :: 60
WIDTH :: 800
HEIGHT :: 600

ENABLE_VALIDATION_LAYERS :: #config(ENABLE_VALIDATION_LAYERS, ODIN_DEBUG)

SHADER_BASIC_VERT :: #load("./shaders/bin/basic.vert.spv")
SHADER_BASIC_FRAG :: #load("./shaders/bin/basic.frag.spv")

MAX_FRAMES_IN_FLIGHT :: 2

MODEL_PATH :: "textures/viking_room.obj"
TEXTURE_PATH :: "textures/viking_room.png"

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
	// instance
	window:                   glfw.WindowHandle,
	instance:                 vk.Instance,
	debugMessenger:           vk.DebugUtilsMessengerEXT,
	surface:                  vk.SurfaceKHR,
	physicalDevice:           vk.PhysicalDevice,
	msaaSamples:              vk.SampleCountFlags,
	device:                   vk.Device,

	// queues and swapchains
	graphicsQueue:            vk.Queue,
	presentQueue:             vk.Queue,
	swapChain:                vk.SwapchainKHR,
	swapChainImages:          []vk.Image,
	swapChainImageFormat:     vk.Format,
	swapChainExtent:          vk.Extent2D,
	swapChainImageViews:      []vk.ImageView,
	swapChainFramebuffers:    []vk.Framebuffer,

	// renderpass and pipeline
	renderPass:               vk.RenderPass,
	descriptorSetLayout:      vk.DescriptorSetLayout,
	descriptorPool:           vk.DescriptorPool,
	descriptorSets:           []vk.DescriptorSet,
	pipelineLayout:           vk.PipelineLayout,
	graphicsPipeline:         vk.Pipeline,
	commandPool:              vk.CommandPool,

	// buffers
	colorImage:               vk.Image,
	colorImageMemory:         vk.DeviceMemory,
	colorImageView:           vk.ImageView,
	depthImage:               vk.Image,
	depthImageMemory:         vk.DeviceMemory,
	depthImageView:           vk.ImageView,
	mipLevels:                u32,
	textureImage:             vk.Image,
	textureImageMemory:       vk.DeviceMemory,
	textureImageView:         vk.ImageView,
	textureSampler:           vk.Sampler,
	vertices:                 [dynamic]Vertex,
	indices:                  [dynamic]u32,
	vertexBuffer:             vk.Buffer,
	vertexBufferMemory:       vk.DeviceMemory,
	indexBuffer:              vk.Buffer,
	indexBufferMemory:        vk.DeviceMemory,
	uniformBuffers:           []vk.Buffer,
	uniformBuffersMemory:     []vk.DeviceMemory,
	uniformBuffersMapped:     []rawptr,
	commandBuffers:           []vk.CommandBuffer,

	// sync objects
	imageAvailableSemaphores: []vk.Semaphore,
	renderFinishedSemaphores: []vk.Semaphore,
	inFlightFences:           []vk.Fence,
	imagesInFlight:           []vk.Fence,
	currentFrame:             int,
	framebufferResized:       bool,

	// game objects
	startTime:                time.Tick,
	frameCounter:             i64,
	frame_start:              time.Tick,
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

	app.msaaSamples = {._1}

	setupDebugMessenger(app)
	createSurface(app)
	pickPhysicalDevice(app)
	createLogicalDevice(app)
	createSwapChain(app)
	createImageViews(app)
	createRenderPass(app)
	createDescriptorSetLayout(app)
	createGraphicsPipeline(app)
	createCommandPool(app)
	createColorResources(app)
	createDepthResources(app)
	createFramebuffers(app)
	createTextureImage(app)
	createTextureImageView(app)
	createTextureSampler(app)
	loadModel(app)
	createVertexBuffer(app)
	createIndexBuffer(app)
	createUniformBuffers(app)
	createDescriptorPool(app)
	createDescriptorSets(app)
	createCommandBuffers(app)
	createSyncObjects(app)
}

beginFrame :: proc(using app: ^HelloTriangleApplication) {
	frame_start = time.tick_now()
}

endFrame :: proc(using app: ^HelloTriangleApplication) {
	frame_time_ms := time.duration_milliseconds(time.tick_since(frame_start))
	remaining_ms := max(0, (1000.0 / FPS) - frame_time_ms)
	time.sleep(time.Millisecond * time.Duration(remaining_ms))
}

mainLoop :: proc(using app: ^HelloTriangleApplication) {
	startTime = time.tick_now()
	for !glfw.WindowShouldClose(window) {
		beginFrame(app)

		glfw.PollEvents()
		if glfw.GetKey(window, glfw.KEY_ESCAPE) == glfw.PRESS {
			glfw.SetWindowShouldClose(window, true)
		}
		drawFrame(app)
		frameCounter += 1

		endFrame(app)
	}
	must(vk.DeviceWaitIdle(device))
}

cleanupSwapChain :: proc(using app: ^HelloTriangleApplication) {
	vk.DestroyImageView(device, depthImageView, nil)
	vk.DestroyImage(device, depthImage, nil)
	vk.FreeMemory(device, depthImageMemory, nil)

	vk.DestroyImageView(device, colorImageView, nil)
	vk.DestroyImage(device, colorImage, nil)
	vk.FreeMemory(device, colorImageMemory, nil)

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

	for i in 0 ..< len(swapChainImages) {
		vk.DestroyBuffer(device, uniformBuffers[i], nil)
		vk.FreeMemory(device, uniformBuffersMemory[i], nil)
	}
	vk.DestroyDescriptorPool(device, descriptorPool, nil)
}

cleanup :: proc(using app: ^HelloTriangleApplication) {
	cleanupSwapChain(app)
	vk.DestroySampler(device, textureSampler, nil)
	vk.DestroyImageView(device, textureImageView, nil)
	vk.DestroyImage(device, textureImage, nil)
	vk.FreeMemory(device, textureImageMemory, nil)
	vk.DestroyDescriptorSetLayout(device, descriptorSetLayout, nil)

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
	delete(uniformBuffers)
	delete(uniformBuffersMemory)
	delete(uniformBuffersMapped)
	delete(descriptorSets)
	delete(vertices)
	delete(indices)
	log.debugf("TOTAL_FRAMES: %v", frameCounter)
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
	createColorResources(app)
	createDepthResources(app)
	createRenderPass(app)
	createGraphicsPipeline(app)
	createFramebuffers(app)
	createUniformBuffers(app)
	createDescriptorPool(app)
	createDescriptorSets(app)
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
	// createInfo.messageType = {.GENERAL, .VALIDATION, .PERFORMANCE}
	createInfo.messageType = {.VALIDATION, .PERFORMANCE}
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
	log.logf(level, "VULKAN: [%s]: %s\n", strings.to_string(b), pCallbackData.pMessage)
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
			physicalDevice = device
			msaaSamples = getMaxUsableSampleCount(app)
			break
		}
	}
	log.assertf(physicalDevice != nil, "failed to find a suitable GPU!")
}

isDeviceSuitable :: proc(surface: vk.SurfaceKHR, device: vk.PhysicalDevice) -> bool {
	props: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(device, &props)
	log.debugf("VULKAN: device: %s type:%v", props.deviceName, props.deviceType)
	if props.deviceType != .DISCRETE_GPU do return false
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
	indices_q := findQueueFamilies(surface, physicalDevice)
	uniqueQueueFamilies: [dynamic]u32
	defer delete(uniqueQueueFamilies)
	append(&uniqueQueueFamilies, indices_q.graphicsFamily.?)
	if uniqueQueueFamilies[0] != indices_q.presentFamily.? {
		append(&uniqueQueueFamilies, indices_q.presentFamily.?)
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
	deviceFeatures.samplerAnisotropy = true

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

	must(vk.CreateDevice(physicalDevice, &createInfo, nil, &device))
	vk.GetDeviceQueue(device, indices_q.graphicsFamily.?, 0, &graphicsQueue)
	vk.GetDeviceQueue(device, indices_q.presentFamily.?, 0, &presentQueue)
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
	swapChainSupport: SwapChainSupportDetails = querySwapChainSupport(surface, physicalDevice)
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

	indices_q := findQueueFamilies(surface, physicalDevice)
	queueFamilyIndices := [?]u32{indices_q.graphicsFamily.?, indices_q.presentFamily.?}

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
		swapChainImageViews[i] = createImageView(
			app,
			swapChainImages[i],
			swapChainImageFormat,
			{.COLOR},
			1,
		)
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
		frontFace               = .COUNTER_CLOCKWISE,
		depthBiasEnable         = false,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo {
		sType                = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		sampleShadingEnable  = false,
		rasterizationSamples = msaaSamples,
	}

	depthStencil := vk.PipelineDepthStencilStateCreateInfo {
		sType                 = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		depthTestEnable       = true,
		depthWriteEnable      = true,
		depthCompareOp        = .LESS,
		depthBoundsTestEnable = false,
		stencilTestEnable     = false,
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
		sType          = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount = 1,
		pSetLayouts    = &descriptorSetLayout,
		// pushConstantRangeCount = 0,
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
		pDepthStencilState  = &depthStencil,
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
		samples        = msaaSamples,
		loadOp         = .CLEAR,
		storeOp        = .STORE,
		stencilLoadOp  = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout  = .UNDEFINED,
		finalLayout    = .COLOR_ATTACHMENT_OPTIMAL,
	}

	depthAttactment := vk.AttachmentDescription {
		format         = findDepthFormat(app),
		samples        = msaaSamples,
		loadOp         = .CLEAR,
		storeOp        = .DONT_CARE,
		stencilLoadOp  = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout  = .UNDEFINED,
		finalLayout    = .DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	}

	colorAttachmentResolve := vk.AttachmentDescription {
		format         = swapChainImageFormat,
		samples        = {._1},
		loadOp         = .DONT_CARE,
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

	depthAttactmentRef := vk.AttachmentReference {
		attachment = 1,
		layout     = .DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	}

	colorAttachmentResolveRef := vk.AttachmentReference {
		attachment = 2,
		layout     = .COLOR_ATTACHMENT_OPTIMAL,
	}

	subpass := vk.SubpassDescription {
		pipelineBindPoint       = .GRAPHICS,
		colorAttachmentCount    = 1,
		pColorAttachments       = &colorAttachmentRef,
		pDepthStencilAttachment = &depthAttactmentRef,
		pResolveAttachments     = &colorAttachmentResolveRef,
	}

	dependency := vk.SubpassDependency {
		srcSubpass    = vk.SUBPASS_EXTERNAL,
		dstSubpass    = 0,
		srcStageMask  = {.COLOR_ATTACHMENT_OUTPUT, .LATE_FRAGMENT_TESTS},
		srcAccessMask = {.COLOR_ATTACHMENT_WRITE, .DEPTH_STENCIL_ATTACHMENT_WRITE},
		dstStageMask  = {.COLOR_ATTACHMENT_OUTPUT, .EARLY_FRAGMENT_TESTS},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE, .DEPTH_STENCIL_ATTACHMENT_WRITE},
	}

	attachments := [?]vk.AttachmentDescription {
		colorAttachment,
		depthAttactment,
		colorAttachmentResolve,
	}

	renderPassInfo := vk.RenderPassCreateInfo {
		sType           = .RENDER_PASS_CREATE_INFO,
		attachmentCount = u32(len(attachments)),
		pAttachments    = raw_data(attachments[:]),
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
		attachments := [?]vk.ImageView{colorImageView, depthImageView, swapChainImageViews[i]}

		framebufferInfo := vk.FramebufferCreateInfo {
			sType           = .FRAMEBUFFER_CREATE_INFO,
			renderPass      = renderPass,
			attachmentCount = u32(len(attachments)),
			pAttachments    = raw_data(attachments[:]),
			width           = swapChainExtent.width,
			height          = swapChainExtent.height,
			layers          = 1,
		}
		must(vk.CreateFramebuffer(device, &framebufferInfo, nil, &swapChainFramebuffers[i]))
	}
}

createCommandPool :: proc(using app: ^HelloTriangleApplication) {
	queueFamilyIndices := findQueueFamilies(surface, physicalDevice)
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

		clearValues := [?]vk.ClearValue {
			{color = vk.ClearColorValue{float32 = [4]f32{0, 0, 0, 1}}},
			{depthStencil = vk.ClearDepthStencilValue{1, 0}},
		}

		renderPassInfo := vk.RenderPassBeginInfo {
			sType = .RENDER_PASS_BEGIN_INFO,
			renderPass = renderPass,
			framebuffer = swapChainFramebuffers[i],
			renderArea = {offset = {0, 0}, extent = swapChainExtent},
			clearValueCount = u32(len(clearValues)),
			pClearValues = raw_data(clearValues[:]),
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

			vk.CmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, .UINT32)
			vk.CmdBindDescriptorSets(
				commandBuffers[i],
				.GRAPHICS,
				pipelineLayout,
				0,
				1,
				&descriptorSets[i],
				0,
				nil,
			)
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
	result := vk.AcquireNextImageKHR(
		device,
		swapChain,
		bits.U64_MAX,
		imageAvailableSemaphores[currentFrame],
		0,
		&imageIndex,
	)
	if result == .ERROR_OUT_OF_DATE_KHR {
		recreateSwapChain(app)
		return
	} else if result != .SUCCESS && result != .SUBOPTIMAL_KHR {
		assert(false, "failed to acquire swap chain image!")
	}

	updateUniformBuffer(app, imageIndex)

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
	pos:      [3]f32,
	color:    [3]f32,
	texCoord: [2]f32,
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
		attributeDescriptions := make([]vk.VertexInputAttributeDescription, 3)
		attributeDescriptions[0] = {
			binding  = 0,
			location = 0,
			format   = .R32G32B32_SFLOAT,
			offset   = u32(offset_of(Vertex, pos)),
		}
		attributeDescriptions[1] = {
			binding  = 0,
			location = 1,
			format   = .R32G32B32_SFLOAT,
			offset   = u32(offset_of(Vertex, color)),
		}
		attributeDescriptions[2] = {
			binding  = 0,
			location = 2,
			format   = .R32G32_SFLOAT,
			offset   = u32(offset_of(Vertex, texCoord)),
		}
		return attributeDescriptions
	}
	unimplemented()
}


UniformBufferObject :: struct #align (16) {
	model, view, proj: glsl.mat4,
}

createVertexBuffer :: proc(using app: ^HelloTriangleApplication) {
	bufferSize := vk.DeviceSize(size_of(vertices[0]) * len(vertices))

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
		size        = size,
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
			app.physicalDevice,
			memRequirements.memoryTypeBits,
			properties,
		),
	}
	must(vk.AllocateMemory(app.device, &allocInfo, nil, bufferMemory))
	must(vk.BindBufferMemory(app.device, buffer^, bufferMemory^, 0))
}

beginSingleTimeCommands :: proc(app: ^HelloTriangleApplication) -> vk.CommandBuffer {
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
	return commandBuffer
}

endSingleTimeCommands :: proc(app: ^HelloTriangleApplication, commandBuffer: ^vk.CommandBuffer) {
	must(vk.EndCommandBuffer(commandBuffer^))

	submitInfo := vk.SubmitInfo {
		sType              = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers    = commandBuffer,
	}

	must(vk.QueueSubmit(app.graphicsQueue, 1, &submitInfo, 0))
	must(vk.QueueWaitIdle(app.graphicsQueue))

	vk.FreeCommandBuffers(app.device, app.commandPool, 1, commandBuffer)
}

copyBuffer :: proc(
	app: ^HelloTriangleApplication,
	srcBuffer, dstBuffer: vk.Buffer,
	size: vk.DeviceSize,
) {
	commandBuffer := beginSingleTimeCommands(app)
	copyRegion := vk.BufferCopy {
		size = size,
	}
	vk.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion)
	endSingleTimeCommands(app, &commandBuffer)
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
	bufferSize := vk.DeviceSize(size_of(indices[0]) * len(indices))

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

createDescriptorSetLayout :: proc(using app: ^HelloTriangleApplication) {
	uboLayoutBinding := vk.DescriptorSetLayoutBinding {
		binding            = 0,
		descriptorCount    = 1,
		descriptorType     = .UNIFORM_BUFFER,
		pImmutableSamplers = nil,
		stageFlags         = {.VERTEX},
	}
	samplerLayoutBinding := vk.DescriptorSetLayoutBinding {
		binding            = 1,
		descriptorCount    = 1,
		descriptorType     = .COMBINED_IMAGE_SAMPLER,
		pImmutableSamplers = nil,
		stageFlags         = {.FRAGMENT},
	}
	bindings := [?]vk.DescriptorSetLayoutBinding{uboLayoutBinding, samplerLayoutBinding}
	layoutInfo := vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		bindingCount = u32(len(bindings)),
		pBindings    = raw_data(bindings[:]),
	}
	must(vk.CreateDescriptorSetLayout(device, &layoutInfo, nil, &descriptorSetLayout))
}

createUniformBuffers :: proc(using app: ^HelloTriangleApplication) {
	bufferSize: vk.DeviceSize = size_of(UniformBufferObject)
	uniformBuffers = make([]vk.Buffer, len(swapChainImages))
	uniformBuffersMemory = make([]vk.DeviceMemory, len(swapChainImages))
	uniformBuffersMapped = make([]rawptr, len(swapChainImages))

	for i in 0 ..< len(swapChainImages) {
		createBuffer(
			app,
			bufferSize,
			{.UNIFORM_BUFFER},
			{.HOST_VISIBLE, .HOST_COHERENT},
			&uniformBuffers[i],
			&uniformBuffersMemory[i],
		)
		must(
			vk.MapMemory(
				device,
				uniformBuffersMemory[i],
				0,
				bufferSize,
				{},
				&uniformBuffersMapped[i],
			),
		)
	}
}

updateUniformBuffer :: proc(app: ^HelloTriangleApplication, currentImage: u32) {
	startTime := app.startTime

	currentTime := time.tick_now()
	duration := f32(time.duration_seconds(time.tick_diff(startTime, currentTime)))
	ubo: UniformBufferObject
	ubo.model = glsl.mat4Rotate({0, 0, 1}, glsl.radians_f32(90) * duration)
	ubo.view = glsl.mat4LookAt({2, 2, 2}, {0, 0, 0}, {0, 0, 1})
	aspect := f32(app.swapChainExtent.width) / f32(app.swapChainExtent.height)
	ubo.proj = glsl.mat4Perspective(glsl.radians_f32(45), aspect, 0.1, 10)

	// glsl is for OpenGL, so flip y axis scale for vulkan
	ubo.proj[1][1] *= -1

	intrinsics.mem_copy_non_overlapping(app.uniformBuffersMapped[currentImage], &ubo, size_of(ubo))
}

createDescriptorPool :: proc(using app: ^HelloTriangleApplication) {
	poolSizes := [?]vk.DescriptorPoolSize {
		{type = .UNIFORM_BUFFER, descriptorCount = u32(len(swapChainImages))},
		{type = .COMBINED_IMAGE_SAMPLER, descriptorCount = u32(len(swapChainImages))},
	}
	poolInfo := vk.DescriptorPoolCreateInfo {
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		poolSizeCount = u32(len(poolSizes)),
		pPoolSizes    = raw_data(poolSizes[:]),
		maxSets       = u32(len(swapChainImages)),
	}
	must(vk.CreateDescriptorPool(device, &poolInfo, nil, &descriptorPool))
}

createDescriptorSets :: proc(using app: ^HelloTriangleApplication) {
	layouts := make([]vk.DescriptorSetLayout, len(swapChainImages))
	defer delete(layouts)
	for _, i in layouts do layouts[i] = descriptorSetLayout

	allocInfo := vk.DescriptorSetAllocateInfo {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool     = descriptorPool,
		descriptorSetCount = u32(len(swapChainImages)),
		pSetLayouts        = raw_data(layouts),
	}

	descriptorSets = make([]vk.DescriptorSet, len(swapChainImages))
	must(vk.AllocateDescriptorSets(device, &allocInfo, raw_data(descriptorSets)))

	for i in 0 ..< len(swapChainImages) {
		bufferInfo := vk.DescriptorBufferInfo {
			buffer = uniformBuffers[i],
			offset = 0,
			range  = size_of(UniformBufferObject),
		}
		imageInfo := vk.DescriptorImageInfo {
			imageLayout = .SHADER_READ_ONLY_OPTIMAL,
			imageView   = textureImageView,
			sampler     = textureSampler,
		}
		descriptorWrites := [?]vk.WriteDescriptorSet {
			{
				sType = .WRITE_DESCRIPTOR_SET,
				dstSet = descriptorSets[i],
				dstBinding = 0,
				dstArrayElement = 0,
				descriptorType = .UNIFORM_BUFFER,
				descriptorCount = 1,
				pBufferInfo = &bufferInfo,
			},
			{
				sType = .WRITE_DESCRIPTOR_SET,
				dstSet = descriptorSets[i],
				dstBinding = 1,
				dstArrayElement = 0,
				descriptorType = .COMBINED_IMAGE_SAMPLER,
				descriptorCount = 1,
				pImageInfo = &imageInfo,
			},
		}
		vk.UpdateDescriptorSets(
			device,
			u32(len(descriptorWrites)),
			raw_data(descriptorWrites[:]),
			0,
			nil,
		)
	}
}


createTextureImage :: proc(using app: ^HelloTriangleApplication) {
	img, err := png.load_from_file(TEXTURE_PATH, {.alpha_add_if_missing}, context.temp_allocator)
	assert(err == nil, fmt.tprintf("image load error: %v", err))
	assert(img.depth == 8)

	texWidth := u32(img.width)
	texHeight := u32(img.height)
	// texChannels := u32(img.channels)
	imageSize := vk.DeviceSize(texWidth * texHeight * 4)
	pixels := img.pixels.buf[:]

	mipLevels = u32(math.floor(math.log2(f32(max(texWidth, texHeight))))) + 1

	stagingBuffer: vk.Buffer
	stagingBufferMemory: vk.DeviceMemory
	createBuffer(
		app,
		imageSize,
		{.TRANSFER_SRC},
		{.HOST_VISIBLE, .HOST_COHERENT},
		&stagingBuffer,
		&stagingBufferMemory,
	)

	data: rawptr
	must(vk.MapMemory(device, stagingBufferMemory, 0, imageSize, {}, &data))
	intrinsics.mem_copy_non_overlapping(data, raw_data(pixels), imageSize)
	vk.UnmapMemory(device, stagingBufferMemory)
	
	// odinfmt: disable
	createImage(app, u32(texWidth), u32(texHeight), mipLevels, {._1}, .R8G8B8A8_SRGB, .OPTIMAL, {.TRANSFER_SRC, .TRANSFER_DST, .SAMPLED}, {.DEVICE_LOCAL}, &textureImage, &textureImageMemory)
	transitionImageLayout(app, textureImage, .R8G8B8A8_SRGB, .UNDEFINED, .TRANSFER_DST_OPTIMAL, mipLevels)
	copyBufferToImage(app, stagingBuffer, textureImage, u32(texWidth), u32(texHeight))
	// transitionImageLayout(app, textureImage, .R8G8B8A8_SRGB, .TRANSFER_DST_OPTIMAL, .SHADER_READ_ONLY_OPTIMAL)
	// transition will happen while generating mipmaps

	vk.DestroyBuffer(device, stagingBuffer, nil)
	vk.FreeMemory(device, stagingBufferMemory, nil)
	generateMipmaps(app, textureImage, .R8G8B8A8_SRGB, texWidth, texHeight, mipLevels)
	// odinfmt: enable
}


createImage :: proc(
	app: ^HelloTriangleApplication,
	width, height, mipLevels: u32,
	numSamples: vk.SampleCountFlags,
	format: vk.Format,
	tiling: vk.ImageTiling,
	usage: vk.ImageUsageFlags,
	properties: vk.MemoryPropertyFlags,
	image: ^vk.Image,
	imageMemory: ^vk.DeviceMemory,
) {
	imageInfo := vk.ImageCreateInfo {
		sType = .IMAGE_CREATE_INFO,
		imageType = .D2,
		extent = {width = width, height = height, depth = 1},
		mipLevels = mipLevels,
		arrayLayers = 1,
		format = format,
		tiling = tiling,
		initialLayout = .UNDEFINED,
		usage = usage,
		samples = numSamples,
		sharingMode = .EXCLUSIVE,
	}
	must(vk.CreateImage(app.device, &imageInfo, nil, image))

	memRequirements: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(app.device, image^, &memRequirements)

	allocInfo := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = memRequirements.size,
		memoryTypeIndex = findMemoryType(
			app.physicalDevice,
			memRequirements.memoryTypeBits,
			properties,
		),
	}
	must(vk.AllocateMemory(app.device, &allocInfo, nil, imageMemory))
	must(vk.BindImageMemory(app.device, image^, imageMemory^, 0))
}

transitionImageLayout :: proc(
	app: ^HelloTriangleApplication,
	image: vk.Image,
	format: vk.Format,
	oldLayout, newLayout: vk.ImageLayout,
	mipLevels: u32,
) {
	commandBuffer := beginSingleTimeCommands(app)

	barrier := vk.ImageMemoryBarrier {
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = oldLayout,
		newLayout = newLayout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = image,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = mipLevels,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}

	sourceStage, destinationStage: vk.PipelineStageFlags
	if oldLayout == .UNDEFINED && newLayout == .TRANSFER_DST_OPTIMAL {
		barrier.srcAccessMask = {}
		barrier.dstAccessMask = {.TRANSFER_WRITE}
		sourceStage = {.TOP_OF_PIPE}
		destinationStage = {.TRANSFER}
	} else if oldLayout == .TRANSFER_DST_OPTIMAL && newLayout == .SHADER_READ_ONLY_OPTIMAL {
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		barrier.dstAccessMask = {.SHADER_READ}
		sourceStage = {.TRANSFER}
		destinationStage = {.FRAGMENT_SHADER}
	} else {
		assert(false, "unsupported")
	}

	vk.CmdPipelineBarrier(
		commandBuffer,
		sourceStage,
		destinationStage,
		{},
		0,
		nil,
		0,
		nil,
		1,
		&barrier,
	)
	endSingleTimeCommands(app, &commandBuffer)
}

copyBufferToImage :: proc(
	app: ^HelloTriangleApplication,
	buffer: vk.Buffer,
	image: vk.Image,
	width, height: u32,
) {
	commandBuffer := beginSingleTimeCommands(app)

	region := vk.BufferImageCopy {
		bufferOffset = 0,
		bufferRowLength = 0,
		bufferImageHeight = 0,
		imageSubresource = {
			aspectMask = {.COLOR},
			mipLevel = 0,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		imageOffset = {0, 0, 0},
		imageExtent = {width, height, 1},
	}
	vk.CmdCopyBufferToImage(commandBuffer, buffer, image, .TRANSFER_DST_OPTIMAL, 1, &region)
	endSingleTimeCommands(app, &commandBuffer)
}

createTextureImageView :: proc(using app: ^HelloTriangleApplication) {
	app.textureImageView = createImageView(app, textureImage, .R8G8B8A8_SRGB, {.COLOR}, mipLevels)
}

createTextureSampler :: proc(using app: ^HelloTriangleApplication) {
	properties: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(physicalDevice, &properties)

	samplerInfo := vk.SamplerCreateInfo {
		sType                   = .SAMPLER_CREATE_INFO,
		magFilter               = .LINEAR,
		minFilter               = .LINEAR,
		addressModeU            = .REPEAT,
		addressModeV            = .REPEAT,
		addressModeW            = .REPEAT,
		anisotropyEnable        = true,
		maxAnisotropy           = properties.limits.maxSamplerAnisotropy,
		borderColor             = .INT_OPAQUE_BLACK,
		unnormalizedCoordinates = false,
		compareEnable           = false,
		compareOp               = .ALWAYS,
		mipmapMode              = .LINEAR,
		minLod                  = 0,
		maxLod                  = vk.LOD_CLAMP_NONE,
		mipLodBias              = 0,
	}
	must(vk.CreateSampler(device, &samplerInfo, nil, &textureSampler))
}

createImageView :: proc(
	app: ^HelloTriangleApplication,
	image: vk.Image,
	format: vk.Format,
	aspectFlags: vk.ImageAspectFlags,
	mipLevels: u32,
) -> vk.ImageView {
	createInfo := vk.ImageViewCreateInfo {
		sType = .IMAGE_VIEW_CREATE_INFO,
		image = image,
		viewType = .D2,
		format = format,
		// components = vk.ComponentMapping {
		// 	r = .IDENTITY,
		// 	g = .IDENTITY,
		// 	b = .IDENTITY,
		// 	a = .IDENTITY,
		// },
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = aspectFlags,
			baseMipLevel = 0,
			levelCount = mipLevels,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}
	imageView: vk.ImageView
	must(vk.CreateImageView(app.device, &createInfo, nil, &imageView))
	return imageView
}

createDepthResources :: proc(using app: ^HelloTriangleApplication) {
	depthFormat := findDepthFormat(app)
	createImage(
		app,
		swapChainExtent.width,
		swapChainExtent.height,
		1,
		msaaSamples,
		depthFormat,
		.OPTIMAL,
		{.DEPTH_STENCIL_ATTACHMENT},
		{.DEVICE_LOCAL},
		&depthImage,
		&depthImageMemory,
	)
	depthImageView = createImageView(app, depthImage, depthFormat, {.DEPTH}, 1)
}

findSupportedFormat :: proc(
	app: ^HelloTriangleApplication,
	candidates: []vk.Format,
	tiling: vk.ImageTiling,
	features: vk.FormatFeatureFlags,
) -> vk.Format {
	for format in candidates {
		props: vk.FormatProperties
		vk.GetPhysicalDeviceFormatProperties(app.physicalDevice, format, &props)
		if tiling == .LINEAR && features <= props.linearTilingFeatures {
			return format
		} else if tiling == .OPTIMAL && features <= props.optimalTilingFeatures {
			return format
		}
	}
	assert(false, "failed to find supported format!")
	return .UNDEFINED
}

findDepthFormat :: proc(app: ^HelloTriangleApplication) -> vk.Format {
	return findSupportedFormat(
		app,
		{.D32_SFLOAT, .D32_SFLOAT_S8_UINT, .D24_UNORM_S8_UINT},
		.OPTIMAL,
		{.DEPTH_STENCIL_ATTACHMENT},
	)
}

hasStencilComponent :: proc(format: vk.Format) -> bool {
	return format == .D32_SFLOAT_S8_UINT || format == .D24_UNORM_S8_UINT
}

loadModel :: proc(using app: ^HelloTriangleApplication) {
	obj_data, ok := os.read_entire_file_from_filename(MODEL_PATH, context.temp_allocator)
	assert(ok)
	// obj := tinyobj.parse_obj(string(obj_data), "", tinyobj.FLAG_TRIANGULATE)
	obj := tinyobj.parse_obj(string(obj_data))
	assert(obj.success)
	defer tinyobj.destroy(&obj)

	uniqueVertices := make(map[Vertex]u32)
	defer delete(uniqueVertices)

	// for shape in obj.shapes {
	// 	for i in 0 ..= shape.length {
	// 		index := obj.attrib.faces[shape.face_offset + i]

	// 		vertex: Vertex
	// 		vertex.pos = {
	// 			obj.attrib.vertices[3 * index.v_idx + 0],
	// 			obj.attrib.vertices[3 * index.v_idx + 1],
	// 			obj.attrib.vertices[3 * index.v_idx + 2],
	// 		}
	// 		vertex.texCoord = {
	// 			obj.attrib.texcoords[2 * index.vt_idx + 0],
	// 			1 - obj.attrib.texcoords[2 * index.vt_idx + 1],
	// 		}
	// 		vertex.color = {1, 1, 1}

	// 		if !(vertex in uniqueVertices) {
	// 			uniqueVertices[vertex] = u32(len(vertices))
	// 			append(&vertices, vertex)
	// 		}
	// 		append(&indices, uniqueVertices[vertex])
	// 	}
	// }

	for index in obj.attrib.faces {
		vertex: Vertex
		vertex.pos = {
			obj.attrib.vertices[3 * index.v_idx + 0],
			obj.attrib.vertices[3 * index.v_idx + 1],
			obj.attrib.vertices[3 * index.v_idx + 2],
		}
		vertex.texCoord = {
			obj.attrib.texcoords[2 * index.vt_idx + 0],
			1 - obj.attrib.texcoords[2 * index.vt_idx + 1],
		}
		vertex.color = {1, 1, 1}

		if !(vertex in uniqueVertices) {
			uniqueVertices[vertex] = u32(len(vertices))
			append(&vertices, vertex)
		}
		append(&indices, uniqueVertices[vertex])
	}

	// tmp_vertices := [?]Vertex {
	// 	{{-0.5, -0.5, 0.0}, {1, 0, 0}, {1, 0}},
	// 	{{0.5, -0.5, 0.0}, {0, 1, 0}, {0, 0}},
	// 	{{0.5, 0.5, 0.0}, {0, 0, 1}, {0, 1}},
	// 	{{-0.5, 0.5, 0.0}, {1, 1, 1}, {1, 1}},
	// 	{{-0.5, -0.5, -0.5}, {1, 0, 0}, {1, 0}},
	// 	{{0.5, -0.5, -0.5}, {0, 1, 0}, {0, 0}},
	// 	{{0.5, 0.5, -0.5}, {0, 0, 1}, {0, 1}},
	// 	{{-0.5, 0.5, -0.5}, {1, 1, 1}, {1, 1}},
	// }

	// tmp_indices := [?]u32{0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4}
	// append_elems(&vertices, ..tmp_vertices[:])
	// append_elems(&indices, ..tmp_indices[:])
}

generateMipmaps :: proc(
	app: ^HelloTriangleApplication,
	image: vk.Image,
	imageFormat: vk.Format,
	texWidth, texHeight, mipLevels: u32,
) {
	formatProperties: vk.FormatProperties
	vk.GetPhysicalDeviceFormatProperties(app.physicalDevice, imageFormat, &formatProperties)

	if !(vk.FormatFeatureFlag.SAMPLED_IMAGE_FILTER_LINEAR in
		   formatProperties.optimalTilingFeatures) {
		assert(false, "texture image format does not support linear blitting!")
	}

	commandBuffer := beginSingleTimeCommands(app)
	barrier := vk.ImageMemoryBarrier {
		sType = .IMAGE_MEMORY_BARRIER,
		image = image,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseArrayLayer = 0,
			layerCount = 1,
			levelCount = 1,
		},
	}

	mipWidth, mipHeight := i32(texWidth), i32(texHeight)
	for i in 1 ..< app.mipLevels {
		barrier.subresourceRange.baseMipLevel = i - 1
		barrier.oldLayout = .TRANSFER_DST_OPTIMAL
		barrier.newLayout = .TRANSFER_SRC_OPTIMAL
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		barrier.dstAccessMask = {.TRANSFER_READ}
		
			// odinfmt: disable
		vk.CmdPipelineBarrier(commandBuffer, {.TRANSFER}, {.TRANSFER}, {}, 0, nil, 0, nil, 1, &barrier)

		blit := vk.ImageBlit {
			srcOffsets = {{0, 0, 0}, {mipWidth, mipHeight, 1}},
			srcSubresource = {
				aspectMask = {.COLOR},
				mipLevel = i - 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
			dstOffsets = {{0, 0, 0}, {max(mipWidth/2, 1), max(mipHeight/2, 1), 1}},
			dstSubresource = {
				aspectMask = {.COLOR},
				mipLevel = i,
				baseArrayLayer = 0,
				layerCount = 1,
			}
		}

		vk.CmdBlitImage(commandBuffer, image, .TRANSFER_SRC_OPTIMAL, image, .TRANSFER_DST_OPTIMAL, 1, &blit, .LINEAR)

		barrier.oldLayout = .TRANSFER_SRC_OPTIMAL
		barrier.newLayout = .SHADER_READ_ONLY_OPTIMAL
		barrier.srcAccessMask = {.TRANSFER_READ}
		barrier.dstAccessMask = {.SHADER_READ}

		vk.CmdPipelineBarrier(commandBuffer, {.TRANSFER}, {.FRAGMENT_SHADER}, {}, 0, nil, 0, nil, 1, &barrier)
		// odinfmt: enable

		if mipWidth > 1 do mipWidth /= 2
		if mipHeight > 1 do mipHeight /= 2
	}

	barrier.subresourceRange.baseMipLevel = mipLevels - 1
	barrier.oldLayout = .TRANSFER_DST_OPTIMAL
	barrier.newLayout = .SHADER_READ_ONLY_OPTIMAL
	barrier.srcAccessMask = {.TRANSFER_WRITE}
	barrier.dstAccessMask = {.SHADER_READ}

	vk.CmdPipelineBarrier(
		commandBuffer,
		{.TRANSFER},
		{.FRAGMENT_SHADER},
		{},
		0,
		nil,
		0,
		nil,
		1,
		&barrier,
	)
	endSingleTimeCommands(app, &commandBuffer)
}

createColorResources :: proc(using app: ^HelloTriangleApplication) {
	colorFormat := swapChainImageFormat
	createImage(
		app,
		swapChainExtent.width,
		swapChainExtent.height,
		1,
		msaaSamples,
		colorFormat,
		.OPTIMAL,
		{.TRANSIENT_ATTACHMENT, .COLOR_ATTACHMENT},
		{.DEVICE_LOCAL},
		&colorImage,
		&colorImageMemory,
	)
	colorImageView = createImageView(app, colorImage, colorFormat, {.COLOR}, 1)
}

getMaxUsableSampleCount :: proc(app: ^HelloTriangleApplication) -> vk.SampleCountFlags {
	physicalDeviceProperties: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(app.physicalDevice, &physicalDeviceProperties)

	counts :=
		physicalDeviceProperties.limits.framebufferColorSampleCounts &
		physicalDeviceProperties.limits.framebufferDepthSampleCounts

	if ._64 in counts do return {._64}
	if ._32 in counts do return {._32}
	if ._16 in counts do return {._16}
	if ._8 in counts do return {._8}
	if ._4 in counts do return {._4}
	if ._2 in counts do return {._2}
	return {._1}
}

