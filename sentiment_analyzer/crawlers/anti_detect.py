"""
反爬对抗模块

提供浏览器指纹伪装、行为模拟、验证码处理和会话管理功能，
用于绕过网站的反爬虫检测机制。
"""

import asyncio
import hashlib
import json
import math
import os
import random
import secrets
import string
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import aiofiles
import aiofiles.os


class BrowserType(str, Enum):
    """浏览器类型枚举"""

    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    OPERA = "opera"


class PlatformType(str, Enum):
    """平台类型枚举"""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    ANDROID = "android"
    IOS = "ios"


class CaptchaType(str, Enum):
    """验证码类型枚举"""

    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    SLIDER = "slider"
    IMAGE_SELECT = "image_select"
    TEXT = "text"


@dataclass
class UserAgentData:
    """
    User-Agent数据结构

    Attributes:
        user_agent: 完整的User-Agent字符串
        browser: 浏览器类型
        browser_version: 浏览器版本
        platform: 平台类型
        platform_version: 平台版本
        engine: 渲染引擎
        engine_version: 引擎版本
    """

    user_agent: str
    browser: BrowserType
    browser_version: str
    platform: PlatformType
    platform_version: str
    engine: str
    engine_version: str


@dataclass
class BrowserFingerprint:
    """
    浏览器指纹数据结构

    Attributes:
        webgl: WebGL指纹
        canvas: Canvas指纹
        audio: 音频指纹
        viewport: 视口大小
        timezone: 时区
        language: 语言
        color_depth: 颜色深度
        device_memory: 设备内存
        hardware_concurrency: CPU核心数
        plugins: 插件列表
        fonts: 字体列表
    """

    webgl: str
    canvas: str
    audio: str
    viewport: tuple[int, int]
    timezone: str
    language: str
    color_depth: int
    device_memory: int
    hardware_concurrency: int
    plugins: list[str] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)


@dataclass
class SessionData:
    """
    会话数据结构

    Attributes:
        session_id: 会话ID
        user_agent: User-Agent数据
        fingerprint: 浏览器指纹
        cookies: Cookie列表
        local_storage: 本地存储
        session_storage: 会话存储
        created_at: 创建时间
        last_used: 最后使用时间
        request_count: 请求次数
        is_valid: 是否有效
    """

    session_id: str
    user_agent: UserAgentData
    fingerprint: BrowserFingerprint
    cookies: list[dict[str, Any]] = field(default_factory=list)
    local_storage: dict[str, str] = field(default_factory=dict)
    session_storage: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    is_valid: bool = True


class UserAgentGenerator:
    """
    User-Agent生成器

    生成随机或特定平台/浏览器的User-Agent字符串，
    支持主流浏览器和操作系统。

    Example:
        >>> generator = UserAgentGenerator()
        >>> ua = generator.get_random()
        >>> ua = generator.get_by_platform(PlatformType.WINDOWS)
        >>> ua = generator.get_by_browser(BrowserType.CHROME)
    """

    _USER_AGENTS: dict[BrowserType, dict[PlatformType, list[str]]] = {
        BrowserType.CHROME: {
            PlatformType.WINDOWS: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
                "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            ],
            PlatformType.MACOS: [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            ],
            PlatformType.LINUX: [
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            ],
            PlatformType.ANDROID: [
                "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Mobile Safari/537.36",
                "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Mobile Safari/537.36",
            ],
        },
        BrowserType.FIREFOX: {
            PlatformType.WINDOWS: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{version}) Gecko/20100101 Firefox/{version}",
            ],
            PlatformType.MACOS: [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:{version}) Gecko/20100101 Firefox/{version}",
            ],
            PlatformType.LINUX: [
                "Mozilla/5.0 (X11; Linux x86_64; rv:{version}) Gecko/20100101 Firefox/{version}",
            ],
            PlatformType.ANDROID: [
                "Mozilla/5.0 (Android 13; Mobile; rv:{version}) Gecko/{version} Firefox/{version}",
            ],
        },
        BrowserType.SAFARI: {
            PlatformType.MACOS: [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Safari/605.1.15",
            ],
            PlatformType.IOS: [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Mobile/15E148 Safari/604.1",
            ],
        },
        BrowserType.EDGE: {
            PlatformType.WINDOWS: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36 Edg/{version}",
            ],
            PlatformType.MACOS: [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36 Edg/{version}",
            ],
        },
        BrowserType.OPERA: {
            PlatformType.WINDOWS: [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36 OPR/{version}",
            ],
            PlatformType.MACOS: [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36 OPR/{version}",
            ],
        },
    }

    _BROWSER_VERSIONS: dict[BrowserType, list[str]] = {
        BrowserType.CHROME: ["120.0.0.0", "121.0.0.0", "122.0.0.0", "123.0.0.0", "124.0.0.0"],
        BrowserType.FIREFOX: ["121.0", "122.0", "123.0", "124.0", "125.0"],
        BrowserType.SAFARI: ["16.0", "16.1", "16.2", "16.3", "17.0"],
        BrowserType.EDGE: ["120.0.0.0", "121.0.0.0", "122.0.0.0", "123.0.0.0"],
        BrowserType.OPERA: ["106.0.0.0", "107.0.0.0", "108.0.0.0", "109.0.0.0"],
    }

    def __init__(self, seed: Optional[int] = None):
        """
        初始化User-Agent生成器

        Args:
            seed: 随机种子，用于可重复生成
        """
        self._rng = random.Random(seed)

    def get_random(self) -> UserAgentData:
        """
        获取随机User-Agent

        Returns:
            随机生成的User-Agent数据
        """
        browser = self._rng.choice(list(BrowserType))
        return self.get_by_browser(browser)

    def get_by_platform(self, platform: PlatformType) -> UserAgentData:
        """
        获取特定平台的User-Agent

        Args:
            platform: 目标平台类型

        Returns:
            符合指定平台的User-Agent数据

        Raises:
            ValueError: 平台不支持任何浏览器
        """
        available_browsers = [
            browser
            for browser, platforms in self._USER_AGENTS.items()
            if platform in platforms
        ]

        if not available_browsers:
            raise ValueError(f"No browsers available for platform: {platform}")

        browser = self._rng.choice(available_browsers)
        return self._generate_user_agent(browser, platform)

    def get_by_browser(self, browser: BrowserType) -> UserAgentData:
        """
        获取特定浏览器的User-Agent

        Args:
            browser: 目标浏览器类型

        Returns:
            符合指定浏览器的User-Agent数据

        Raises:
            ValueError: 浏览器不支持任何平台
        """
        available_platforms = list(self._USER_AGENTS.get(browser, {}).keys())

        if not available_platforms:
            raise ValueError(f"No platforms available for browser: {browser}")

        platform = self._rng.choice(available_platforms)
        return self._generate_user_agent(browser, platform)

    def _generate_user_agent(
        self, browser: BrowserType, platform: PlatformType
    ) -> UserAgentData:
        """
        生成User-Agent数据

        Args:
            browser: 浏览器类型
            platform: 平台类型

        Returns:
            完整的User-Agent数据
        """
        templates = self._USER_AGENTS.get(browser, {}).get(platform, [])
        if not templates:
            raise ValueError(f"No template for {browser} on {platform}")

        template = self._rng.choice(templates)
        version = self._rng.choice(self._BROWSER_VERSIONS.get(browser, ["1.0"]))

        chrome_version = self._rng.choice(
            self._BROWSER_VERSIONS.get(BrowserType.CHROME, ["120.0.0.0"])
        )

        user_agent = template.format(
            version=version,
            chrome_version=chrome_version
        )

        platform_version = self._get_platform_version(platform)

        engine, engine_version = self._get_engine_info(browser, version)

        return UserAgentData(
            user_agent=user_agent,
            browser=browser,
            browser_version=version,
            platform=platform,
            platform_version=platform_version,
            engine=engine,
            engine_version=engine_version,
        )

    def _get_platform_version(self, platform: PlatformType) -> str:
        """获取平台版本"""
        versions = {
            PlatformType.WINDOWS: ["10.0", "11.0"],
            PlatformType.MACOS: ["10.15.7", "11.0", "12.0", "13.0", "14.0"],
            PlatformType.LINUX: ["", "5.15", "6.0"],
            PlatformType.ANDROID: ["12", "13", "14"],
            PlatformType.IOS: ["16.0", "16.5", "17.0", "17.2"],
        }
        return self._rng.choice(versions.get(platform, [""]))

    def _get_engine_info(
        self, browser: BrowserType, version: str
    ) -> tuple[str, str]:
        """获取渲染引擎信息"""
        engine_map = {
            BrowserType.CHROME: ("Blink", version),
            BrowserType.FIREFOX: ("Gecko", version),
            BrowserType.SAFARI: ("WebKit", "605.1.15"),
            BrowserType.EDGE: ("Blink", version),
            BrowserType.OPERA: ("Blink", version),
        }
        return engine_map.get(browser, ("Unknown", "1.0"))


class FingerprintGenerator:
    """
    浏览器指纹生成器

    生成各种浏览器指纹用于绕过指纹追踪检测，
    包括WebGL、Canvas、音频指纹等。

    Example:
        >>> generator = FingerprintGenerator()
        >>> fingerprint = generator.generate()
        >>> webgl = generator.generate_webgl_fingerprint()
        >>> canvas = generator.generate_canvas_fingerprint()
    """

    _WEBGL_RENDERERS = [
        "ANGLE (Intel, Intel(R) UHD Graphics 630, OpenGL 4.1)",
        "ANGLE (NVIDIA, NVIDIA GeForce GTX 1060, OpenGL 4.6)",
        "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070, OpenGL 4.6)",
        "ANGLE (AMD, AMD Radeon RX 580, OpenGL 4.6)",
        "ANGLE (Intel, Intel(R) Iris(R) Xe Graphics, OpenGL 4.1)",
        "ANGLE (Apple, Apple M1, OpenGL 4.1)",
        "ANGLE (NVIDIA, NVIDIA GeForce RTX 4090, OpenGL 4.6)",
        "ANGLE (Intel, Intel(R) HD Graphics 4000, OpenGL 4.1)",
    ]

    _WEBGL_VENDORS = [
        "Google Inc. (Intel)",
        "Google Inc. (NVIDIA)",
        "Google Inc. (AMD)",
        "Google Inc. (Apple)",
        "Intel Inc.",
        "NVIDIA Corporation",
        "AMD",
    ]

    _VIEWPORT_SIZES = [
        (1920, 1080),
        (1920, 1200),
        (2560, 1440),
        (1366, 768),
        (1536, 864),
        (1440, 900),
        (1680, 1050),
        (2560, 1080),
        (3840, 2160),
        (1280, 720),
    ]

    _TIMEZONES = [
        "America/New_York",
        "America/Los_Angeles",
        "America/Chicago",
        "Europe/London",
        "Europe/Paris",
        "Europe/Berlin",
        "Asia/Tokyo",
        "Asia/Shanghai",
        "Asia/Singapore",
        "Australia/Sydney",
    ]

    _LANGUAGES = [
        "en-US",
        "en-GB",
        "zh-CN",
        "zh-TW",
        "ja-JP",
        "ko-KR",
        "de-DE",
        "fr-FR",
        "es-ES",
        "pt-BR",
    ]

    _COLOR_DEPTHS = [24, 32]

    _DEVICE_MEMORIES = [4, 8, 16, 32]

    _HARDWARE_CONCURRENCIES = [4, 6, 8, 12, 16, 24, 32]

    def __init__(self, seed: Optional[int] = None):
        """
        初始化指纹生成器

        Args:
            seed: 随机种子，用于可重复生成
        """
        self._rng = random.Random(seed)

    def generate(self) -> BrowserFingerprint:
        """
        生成完整的浏览器指纹

        Returns:
            完整的浏览器指纹数据
        """
        return BrowserFingerprint(
            webgl=self.generate_webgl_fingerprint(),
            canvas=self.generate_canvas_fingerprint(),
            audio=self.generate_audio_fingerprint(),
            viewport=self.get_random_viewport(),
            timezone=self.get_random_timezone(),
            language=self._rng.choice(self._LANGUAGES),
            color_depth=self._rng.choice(self._COLOR_DEPTHS),
            device_memory=self._rng.choice(self._DEVICE_MEMORIES),
            hardware_concurrency=self._rng.choice(self._HARDWARE_CONCURRENCIES),
            plugins=self._generate_plugins(),
            fonts=self._generate_fonts(),
        )

    def generate_webgl_fingerprint(self) -> str:
        """
        生成WebGL指纹

        基于渲染器和厂商信息生成唯一的WebGL指纹。

        Returns:
            WebGL指纹字符串
        """
        vendor = self._rng.choice(self._WEBGL_VENDORS)
        renderer = self._rng.choice(self._WEBGL_RENDERERS)

        extensions = self._generate_webgl_extensions()
        parameters = self._generate_webgl_parameters()

        fingerprint_data = f"{vendor}|{renderer}|{extensions}|{parameters}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:32]

    def generate_canvas_fingerprint(self) -> str:
        """
        生成Canvas指纹

        基于Canvas渲染特性生成唯一指纹。

        Returns:
            Canvas指纹字符串
        """
        noise = "".join(
            self._rng.choices(string.ascii_letters + string.digits, k=64)
        )

        canvas_data = f"canvas_{noise}_{self._rng.random()}"
        return hashlib.md5(canvas_data.encode()).hexdigest()

    def generate_audio_fingerprint(self) -> str:
        """
        生成音频指纹

        基于音频处理特性生成唯一指纹。

        Returns:
            音频指纹字符串
        """
        sample_rate = self._rng.choice([44100, 48000])
        channel_count = self._rng.choice([2, 6])
        latency = self._rng.uniform(0.01, 0.05)

        audio_data = f"audio_{sample_rate}_{channel_count}_{latency:.6f}"
        return hashlib.sha256(audio_data.encode()).hexdigest()[:32]

    def get_random_viewport(self) -> tuple[int, int]:
        """
        获取随机视口大小

        Returns:
            视口宽度和高度的元组
        """
        return self._rng.choice(self._VIEWPORT_SIZES)

    def get_random_timezone(self) -> str:
        """
        获取随机时区

        Returns:
            时区字符串
        """
        return self._rng.choice(self._TIMEZONES)

    def _generate_webgl_extensions(self) -> str:
        """生成WebGL扩展列表"""
        extensions = [
            "ANGLE_instanced_arrays",
            "EXT_blend_minmax",
            "EXT_color_buffer_half_float",
            "EXT_float_blend",
            "EXT_frag_depth",
            "EXT_shader_texture_lod",
            "EXT_texture_compression_bptc",
            "EXT_texture_compression_rgtc",
            "EXT_texture_filter_anisotropic",
            "WEBKIT_EXT_texture_filter_anisotropic",
            "EXT_texture_norm16",
            "OES_element_index_uint",
            "OES_fbo_render_mipmap",
            "OES_standard_derivatives",
            "OES_texture_float",
            "OES_texture_float_linear",
            "OES_texture_half_float",
            "OES_texture_half_float_linear",
            "OES_vertex_array_object",
            "WEBGL_color_buffer_float",
            "WEBGL_compressed_texture_s3tc",
            "WEBKIT_WEBGL_compressed_texture_s3tc",
            "WEBGL_compressed_texture_s3tc_srgb",
            "WEBGL_debug_renderer_info",
            "WEBGL_debug_shaders",
            "WEBGL_depth_texture",
            "WEBKIT_WEBGL_depth_texture",
            "WEBGL_draw_buffers",
            "WEBGL_lose_context",
            "WEBKIT_WEBGL_lose_context",
            "WEBGL_multi_draw",
        ]
        count = self._rng.randint(15, len(extensions))
        return ",".join(self._rng.sample(extensions, count))

    def _generate_webgl_parameters(self) -> str:
        """生成WebGL参数"""
        params = {
            "MAX_TEXTURE_SIZE": self._rng.choice([8192, 16384, 32768]),
            "MAX_VERTEX_ATTRIBS": self._rng.randint(16, 64),
            "MAX_VERTEX_UNIFORM_VECTORS": self._rng.randint(256, 4096),
            "MAX_VARYING_VECTORS": self._rng.randint(8, 64),
            "MAX_COMBINED_TEXTURE_IMAGE_UNITS": self._rng.randint(32, 192),
            "MAX_VERTEX_TEXTURE_IMAGE_UNITS": self._rng.randint(4, 32),
            "MAX_TEXTURE_IMAGE_UNITS": self._rng.randint(8, 32),
            "MAX_FRAGMENT_UNIFORM_VECTORS": self._rng.randint(256, 1024),
        }
        return str(params)

    def _generate_plugins(self) -> list[str]:
        """生成插件列表"""
        all_plugins = [
            "PDF Viewer",
            "Chrome PDF Viewer",
            "Chromium PDF Viewer",
            "Microsoft Edge PDF Viewer",
            "WebKit built-in PDF",
            "Widevine Content Decryption Module",
            "Native Client",
        ]
        count = self._rng.randint(2, len(all_plugins))
        return self._rng.sample(all_plugins, count)

    def _generate_fonts(self) -> list[str]:
        """生成字体列表"""
        all_fonts = [
            "Arial",
            "Arial Black",
            "Comic Sans MS",
            "Courier New",
            "Georgia",
            "Impact",
            "Times New Roman",
            "Trebuchet MS",
            "Verdana",
            "Helvetica",
            "Palatino",
            "Garamond",
            "Bookman",
            "Avant Garde",
            "Candara",
            "Calibri",
            "Cambria",
            "Consolas",
            "Monaco",
            "Lucida Console",
        ]
        count = self._rng.randint(8, len(all_fonts))
        return self._rng.sample(all_fonts, count)


class BehaviorSimulator:
    """
    行为模拟器

    模拟人类用户的行为模式，包括鼠标移动、滚动、打字等，
    使用泊松分布等统计模型生成自然的延迟和间隔。

    Example:
        >>> simulator = BehaviorSimulator()
        >>> await simulator.random_delay(1.0, 5.0)
        >>> movements = simulator.simulate_mouse_movement()
        >>> intervals = simulator.get_human_like_intervals(10)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化行为模拟器

        Args:
            seed: 随机种子，用于可重复生成
        """
        self._rng = random.Random(seed)
        self._last_mouse_pos: tuple[float, float] = (0.0, 0.0)
        self._last_action_time: float = time.monotonic()

    async def random_delay(
        self,
        min_seconds: float,
        max_seconds: float,
        use_poisson: bool = True
    ) -> float:
        """
        随机延迟

        使用泊松分布或均匀分布生成随机延迟时间。

        Args:
            min_seconds: 最小延迟秒数
            max_seconds: 最大延迟秒数
            use_poisson: 是否使用泊松分布

        Returns:
            实际延迟的秒数
        """
        if use_poisson:
            lambda_param = (min_seconds + max_seconds) / 2
            delay = self._rng.expovariate(1 / lambda_param)
            delay = max(min_seconds, min(max_seconds, delay))
        else:
            delay = self._rng.uniform(min_seconds, max_seconds)

        await asyncio.sleep(delay)
        return delay

    def simulate_mouse_movement(
        self,
        start: Optional[tuple[float, float]] = None,
        end: Optional[tuple[float, float]] = None,
        steps: Optional[int] = None,
    ) -> list[tuple[float, float]]:
        """
        模拟鼠标移动

        生成从起点到终点的自然鼠标移动轨迹，使用贝塞尔曲线。

        Args:
            start: 起始坐标，默认为上次位置
            end: 终点坐标，默认为随机位置
            steps: 移动步数，默认为随机值

        Returns:
            鼠标移动轨迹坐标列表
        """
        if start is None:
            start = self._last_mouse_pos

        if end is None:
            end = (
                self._rng.uniform(100, 1800),
                self._rng.uniform(100, 900)
            )

        if steps is None:
            distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            steps = max(5, int(distance / 10))

        control_points = self._generate_bezier_control_points(start, end)

        trajectory: list[tuple[float, float]] = []
        for i in range(steps + 1):
            t = i / steps
            t = self._apply_easing(t, "ease_in_out")
            point = self._bezier_point(control_points, t)
            noise_x = self._rng.gauss(0, 1)
            noise_y = self._rng.gauss(0, 1)
            trajectory.append((
                point[0] + noise_x,
                point[1] + noise_y
            ))

        self._last_mouse_pos = end
        return trajectory

    def simulate_scroll(
        self,
        distance: Optional[int] = None,
        direction: str = "down"
    ) -> list[tuple[int, int]]:
        """
        模拟滚动

        生成自然的滚动轨迹。

        Args:
            distance: 滚动距离（像素），默认为随机值
            direction: 滚动方向，"up" 或 "down"

        Returns:
            滚动轨迹列表，每个元素为(位置, 时间戳)
        """
        if distance is None:
            distance = self._rng.randint(200, 1000)

        sign = 1 if direction == "down" else -1

        scroll_events: list[tuple[int, int]] = []
        current_pos = 0
        total_time = 0

        while abs(current_pos) < distance:
            delta = self._rng.randint(20, 100)
            delta = min(delta, distance - abs(current_pos))
            current_pos += sign * delta

            time_delta = self._rng.randint(50, 200)
            total_time += time_delta

            scroll_events.append((current_pos, total_time))

        return scroll_events

    def simulate_typing(
        self,
        text: str,
        wpm: Optional[int] = None
    ) -> list[tuple[str, float]]:
        """
        模拟打字

        生成自然的打字时间间隔。

        Args:
            text: 要输入的文本
            wpm: 每分钟字数，默认为随机值（40-80）

        Returns:
            打字事件列表，每个元素为(字符, 时间戳)
        """
        if wpm is None:
            wpm = self._rng.randint(40, 80)

        base_interval = 60.0 / (wpm * 5)

        typing_events: list[tuple[str, float]] = []
        total_time = 0.0

        for i, char in enumerate(text):
            if char == " ":
                interval = base_interval * self._rng.uniform(0.8, 1.2)
            elif char in ".!?":
                interval = base_interval * self._rng.uniform(2.0, 4.0)
            elif char == ",":
                interval = base_interval * self._rng.uniform(1.5, 2.5)
            elif char in "({[":
                interval = base_interval * self._rng.uniform(1.2, 1.8)
            else:
                interval = base_interval * self._rng.uniform(0.7, 1.5)

            if self._rng.random() < 0.02:
                interval += self._rng.uniform(0.3, 0.8)

            total_time += interval
            typing_events.append((char, total_time))

        return typing_events

    def get_human_like_intervals(
        self,
        count: int,
        min_interval: float = 0.5,
        max_interval: float = 5.0
    ) -> list[float]:
        """
        获取类人时间间隔序列

        使用泊松过程生成自然的时间间隔序列。

        Args:
            count: 需要的间隔数量
            min_interval: 最小间隔
            max_interval: 最大间隔

        Returns:
            时间间隔列表（秒）
        """
        lambda_param = (min_interval + max_interval) / 2
        intervals: list[float] = []

        for _ in range(count):
            interval = self._rng.expovariate(1 / lambda_param)
            interval = max(min_interval, min(max_interval, interval))
            intervals.append(interval)

        return intervals

    def _generate_bezier_control_points(
        self,
        start: tuple[float, float],
        end: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """生成贝塞尔曲线控制点"""
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        offset_x = self._rng.uniform(-200, 200)
        offset_y = self._rng.uniform(-200, 200)

        control1 = (mid_x + offset_x, mid_y + offset_y)
        control2 = (mid_x - offset_x, mid_y - offset_y)

        return [start, control1, control2, end]

    def _bezier_point(
        self,
        points: list[tuple[float, float]],
        t: float
    ) -> tuple[float, float]:
        """计算贝塞尔曲线上的点"""
        n = len(points) - 1
        x = sum(
            self._binomial(n, i) * (1 - t)**(n - i) * t**i * points[i][0]
            for i in range(n + 1)
        )
        y = sum(
            self._binomial(n, i) * (1 - t)**(n - i) * t**i * points[i][1]
            for i in range(n + 1)
        )
        return (x, y)

    def _binomial(self, n: int, k: int) -> int:
        """计算二项式系数"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    def _apply_easing(self, t: float, easing_type: str) -> float:
        """应用缓动函数"""
        if easing_type == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - (-2 * t + 2)**2 / 2
        elif easing_type == "ease_in":
            return t * t
        elif easing_type == "ease_out":
            return 1 - (1 - t)**2
        return t


class CaptchaHandler:
    """
    验证码处理器

    处理各种类型的验证码，包括reCAPTCHA、hCaptcha、滑块验证码等，
    支持第三方打码平台接口。

    Example:
        >>> handler = CaptchaHandler(api_key="your_api_key")
        >>> result = await handler.handle_recaptcha(page, site_key)
        >>> result = await handler.handle_slider_captcha(page)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        service_url: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        初始化验证码处理器

        Args:
            api_key: 第三方打码平台API密钥
            service_url: 打码服务URL
            timeout: 超时时间（秒）
        """
        self._api_key = api_key
        self._service_url = service_url or "https://api.2captcha.com"
        self._timeout = timeout
        self._rng = random.Random()

    async def handle_recaptcha(
        self,
        page: Any,
        site_key: Optional[str] = None,
        action: Optional[str] = None,
    ) -> bool:
        """
        处理reCAPTCHA

        Args:
            page: Playwright页面对象
            site_key: 站点密钥（可选，自动检测）
            action: reCAPTCHA v3动作名称

        Returns:
            是否成功处理
        """
        try:
            if site_key is None:
                site_key = await self._extract_recaptcha_site_key(page)

            if site_key is None:
                return False

            captcha_type = await self._detect_recaptcha_version(page)

            if captcha_type == CaptchaType.RECAPTCHA_V3:
                return await self._handle_recaptcha_v3(page, site_key, action)
            else:
                return await self._handle_recaptcha_v2(page, site_key)

        except Exception:
            return False

    async def handle_hcaptcha(
        self,
        page: Any,
        site_key: Optional[str] = None,
    ) -> bool:
        """
        处理hCaptcha

        Args:
            page: Playwright页面对象
            site_key: 站点密钥（可选，自动检测）

        Returns:
            是否成功处理
        """
        try:
            if site_key is None:
                site_key = await self._extract_hcaptcha_site_key(page)

            if site_key is None:
                return False

            if self._api_key:
                token = await self._solve_via_service(
                    CaptchaType.HCAPTCHA,
                    {"sitekey": site_key, "pageurl": page.url}
                )
                if token:
                    await self._inject_hcaptcha_token(page, token)
                    return True

            return await self._simulate_hcaptcha_click(page)

        except Exception:
            return False

    async def handle_slider_captcha(
        self,
        page: Any,
        slider_selector: Optional[str] = None,
    ) -> bool:
        """
        处理滑块验证码

        Args:
            page: Playwright页面对象
            slider_selector: 滑块选择器（可选，自动检测）

        Returns:
            是否成功处理
        """
        try:
            if slider_selector is None:
                slider_selector = await self._detect_slider_selector(page)

            if slider_selector is None:
                return False

            slider = await page.query_selector(slider_selector)
            if slider is None:
                return False

            box = await slider.bounding_box()
            if box is None:
                return False

            gap_position = await self._detect_slider_gap(page)
            if gap_position is None:
                gap_position = self._rng.randint(100, 300)

            trajectory = self._generate_slider_trajectory(gap_position)

            await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            await page.mouse.down()

            current_x = box["x"] + box["width"] / 2
            current_y = box["y"] + box["height"] / 2

            for x, y, delay in trajectory:
                await page.mouse.move(current_x + x, current_y + y)
                current_x += x
                await asyncio.sleep(delay / 1000)

            await asyncio.sleep(self._rng.uniform(0.1, 0.3))
            await page.mouse.up()

            await asyncio.sleep(1)
            return True

        except Exception:
            return False

    async def solve_image_captcha(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        解决图片验证码

        Args:
            image_data: 图片二进制数据
            prompt: 提示文本（可选）

        Returns:
            验证码文本
        """
        if not self._api_key:
            return None

        try:
            return await self._solve_via_service(
                CaptchaType.IMAGE_SELECT,
                {"image": image_data, "prompt": prompt}
            )
        except Exception:
            return None

    async def _extract_recaptcha_site_key(self, page: Any) -> Optional[str]:
        """提取reCAPTCHA站点密钥"""
        selectors = [
            '[data-sitekey]',
            '.g-recaptcha',
            '#g-recaptcha-response',
        ]

        for selector in selectors:
            element = await page.query_selector(selector)
            if element:
                site_key = await element.get_attribute('data-sitekey')
                if site_key:
                    return site_key

        content = await page.content()
        import re
        match = re.search(r'data-sitekey=["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)

        return None

    async def _extract_hcaptcha_site_key(self, page: Any) -> Optional[str]:
        """提取hCaptcha站点密钥"""
        selectors = [
            '[data-sitekey]',
            '.h-captcha',
        ]

        for selector in selectors:
            element = await page.query_selector(selector)
            if element:
                site_key = await element.get_attribute('data-sitekey')
                if site_key:
                    return site_key

        return None

    async def _detect_recaptcha_version(self, page: Any) -> CaptchaType:
        """检测reCAPTCHA版本"""
        v3_script = await page.query_selector('script[src*="recaptcha/enterprise"]')
        if v3_script:
            return CaptchaType.RECAPTCHA_V3
        return CaptchaType.RECAPTCHA_V2

    async def _handle_recaptcha_v2(
        self,
        page: Any,
        site_key: str,
    ) -> bool:
        """处理reCAPTCHA v2"""
        if self._api_key:
            token = await self._solve_via_service(
                CaptchaType.RECAPTCHA_V2,
                {"sitekey": site_key, "pageurl": page.url}
            )
            if token:
                await page.evaluate(f'''
                    document.getElementById('g-recaptcha-response').innerHTML = '{token}';
                ''')
                return True

        checkbox = await page.query_selector('.recaptcha-checkbox')
        if checkbox:
            await checkbox.click()
            await asyncio.sleep(2)

            if await self._check_recaptcha_solved(page):
                return True

        return False

    async def _handle_recaptcha_v3(
        self,
        page: Any,
        site_key: str,
        action: Optional[str] = None,
    ) -> bool:
        """处理reCAPTCHA v3"""
        if self._api_key:
            token = await self._solve_via_service(
                CaptchaType.RECAPTCHA_V3,
                {
                    "sitekey": site_key,
                    "pageurl": page.url,
                    "action": action or "submit"
                }
            )
            if token:
                await page.evaluate(f'''
                    if (window.grecaptcha) {{
                        grecaptcha.execute('{site_key}', {{action: '{action or "submit"}'}})
                            .then(function(token) {{
                                document.getElementById('g-recaptcha-response').value = token;
                            }});
                    }}
                ''')
                return True

        return False

    async def _simulate_hcaptcha_click(self, page: Any) -> bool:
        """模拟hCaptcha点击"""
        checkbox = await page.query_selector('#checkbox')
        if checkbox:
            await checkbox.click()
            await asyncio.sleep(2)
            return True
        return False

    async def _inject_hcaptcha_token(self, page: Any, token: str) -> None:
        """注入hCaptcha令牌"""
        await page.evaluate(f'''
            if (window.hcaptcha) {{
                hcaptcha.setResponse('{token}');
            }}
        ''')

    async def _detect_slider_selector(self, page: Any) -> Optional[str]:
        """检测滑块选择器"""
        selectors = [
            '.slider',
            '[class*="slider"]',
            '[class*="drag"]',
            '.slide-block',
            '.tcaptcha-slider',
        ]

        for selector in selectors:
            element = await page.query_selector(selector)
            if element:
                return selector

        return None

    async def _detect_slider_gap(self, page: Any) -> Optional[int]:
        """检测滑块缺口位置"""
        try:
            gap = await page.evaluate('''
                () => {
                    const gapElement = document.querySelector('.slider-gap, .slide-gap, [class*="gap"]');
                    if (gapElement) {
                        return gapElement.offsetLeft;
                    }
                    return null;
                }
            ''')
            return gap
        except Exception:
            return None

    def _generate_slider_trajectory(
        self,
        distance: int,
    ) -> list[tuple[int, int, int]]:
        """
        生成滑块轨迹

        Args:
            distance: 滑动距离

        Returns:
            轨迹列表，每个元素为(x偏移, y偏移, 延迟毫秒)
        """
        trajectory: list[tuple[int, int, int]] = []

        current = 0
        while current < distance:
            remaining = distance - current
            if remaining < 10:
                step = remaining
            else:
                step = self._rng.randint(5, min(20, remaining))

            y_offset = self._rng.randint(-2, 2)
            delay = self._rng.randint(5, 20)

            trajectory.append((step, y_offset, delay))
            current += step

        if self._rng.random() < 0.3:
            overshoot = self._rng.randint(5, 15)
            trajectory.append((-overshoot, self._rng.randint(-1, 1), self._rng.randint(50, 100)))

        return trajectory

    async def _check_recaptcha_solved(self, page: Any) -> bool:
        """检查reCAPTCHA是否已解决"""
        try:
            checked = await page.evaluate('''
                () => {
                    const checkbox = document.querySelector('.recaptcha-checkbox');
                    return checkbox && checkbox.classList.contains('recaptcha-checkbox-checked');
                }
            ''')
            return bool(checked)
        except Exception:
            return False

    async def _solve_via_service(
        self,
        captcha_type: CaptchaType,
        params: dict[str, Any],
    ) -> Optional[str]:
        """
        通过第三方服务解决验证码

        Args:
            captcha_type: 验证码类型
            params: 参数字典

        Returns:
            验证码解决方案
        """
        if not self._api_key:
            return None

        return None


class SessionManager:
    """
    会话管理器

    管理爬虫会话的创建、保存、加载和轮换，
    支持Cookie持久化和会话有效性检测。

    Example:
        >>> manager = SessionManager(storage_dir="./sessions")
        >>> session = await manager.create_session()
        >>> await manager.save_cookies(session.session_id, cookies)
        >>> loaded = await manager.load_cookies(session.session_id)
        >>> is_valid = await manager.check_session_valid(session)
    """

    def __init__(
        self,
        storage_dir: str = "./data/sessions",
        max_sessions: int = 100,
        session_ttl: int = 3600,
    ):
        """
        初始化会话管理器

        Args:
            storage_dir: 会话存储目录
            max_sessions: 最大会话数
            session_ttl: 会话生存时间（秒）
        """
        self._storage_dir = Path(storage_dir)
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl
        self._sessions: dict[str, SessionData] = {}
        self._ua_generator = UserAgentGenerator()
        self._fp_generator = FingerprintGenerator()
        self._rng = random.Random()

    async def initialize(self) -> None:
        """初始化会话管理器，创建存储目录"""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        await self._load_existing_sessions()

    async def create_session(
        self,
        user_agent: Optional[UserAgentData] = None,
        fingerprint: Optional[BrowserFingerprint] = None,
    ) -> SessionData:
        """
        创建新会话

        Args:
            user_agent: 自定义User-Agent（可选）
            fingerprint: 自定义浏览器指纹（可选）

        Returns:
            新创建的会话数据
        """
        if len(self._sessions) >= self._max_sessions:
            await self._cleanup_old_sessions()

        session_id = self._generate_session_id()

        session = SessionData(
            session_id=session_id,
            user_agent=user_agent or self._ua_generator.get_random(),
            fingerprint=fingerprint or self._fp_generator.generate(),
        )

        self._sessions[session_id] = session
        await self._save_session(session)

        return session

    async def save_cookies(
        self,
        session_id: str,
        cookies: list[dict[str, Any]],
    ) -> bool:
        """
        保存Cookie

        Args:
            session_id: 会话ID
            cookies: Cookie列表

        Returns:
            是否保存成功
        """
        session = self._sessions.get(session_id)
        if session is None:
            return False

        session.cookies = cookies
        session.last_used = datetime.utcnow()
        await self._save_session(session)

        return True

    async def load_cookies(
        self,
        session_id: str,
    ) -> Optional[list[dict[str, Any]]]:
        """
        加载Cookie

        Args:
            session_id: 会话ID

        Returns:
            Cookie列表，如果会话不存在则返回None
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        session.last_used = datetime.utcnow()
        return session.cookies

    async def rotate_session(
        self,
        old_session_id: Optional[str] = None,
    ) -> SessionData:
        """
        轮换会话

        创建新会话并可选择性地使旧会话失效。

        Args:
            old_session_id: 要失效的旧会话ID（可选）

        Returns:
            新创建的会话数据
        """
        if old_session_id and old_session_id in self._sessions:
            old_session = self._sessions[old_session_id]
            old_session.is_valid = False
            await self._save_session(old_session)

        return await self.create_session()

    async def check_session_valid(
        self,
        session: Union[str, SessionData],
    ) -> bool:
        """
        检查会话有效性

        Args:
            session: 会话ID或会话数据

        Returns:
            会话是否有效
        """
        if isinstance(session, str):
            session_data = self._sessions.get(session)
            if session_data is None:
                return False
        else:
            session_data = session

        if not session_data.is_valid:
            return False

        age = (datetime.utcnow() - session_data.created_at).total_seconds()
        if age > self._session_ttl:
            session_data.is_valid = False
            return False

        return True

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        获取会话

        Args:
            session_id: 会话ID

        Returns:
            会话数据，如果不存在则返回None
        """
        return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        if session_id not in self._sessions:
            return False

        del self._sessions[session_id]

        session_file = self._storage_dir / f"{session_id}.json"
        if session_file.exists():
            await aiofiles.os.remove(session_file)

        return True

    async def get_active_sessions(self) -> list[SessionData]:
        """
        获取所有活跃会话

        Returns:
            活跃会话列表
        """
        active = []
        for session in self._sessions.values():
            if await self.check_session_valid(session):
                active.append(session)
        return active

    async def update_session_usage(
        self,
        session_id: str,
        increment_requests: bool = True,
    ) -> bool:
        """
        更新会话使用状态

        Args:
            session_id: 会话ID
            increment_requests: 是否增加请求计数

        Returns:
            是否更新成功
        """
        session = self._sessions.get(session_id)
        if session is None:
            return False

        session.last_used = datetime.utcnow()
        if increment_requests:
            session.request_count += 1

        await self._save_session(session)
        return True

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        timestamp = int(time.time())
        random_part = secrets.token_hex(8)
        return f"sess_{timestamp}_{random_part}"

    async def _save_session(self, session: SessionData) -> None:
        """保存会话到文件"""
        session_file = self._storage_dir / f"{session.session_id}.json"

        data = {
            "session_id": session.session_id,
            "user_agent": {
                "user_agent": session.user_agent.user_agent,
                "browser": session.user_agent.browser.value,
                "browser_version": session.user_agent.browser_version,
                "platform": session.user_agent.platform.value,
                "platform_version": session.user_agent.platform_version,
                "engine": session.user_agent.engine,
                "engine_version": session.user_agent.engine_version,
            },
            "fingerprint": {
                "webgl": session.fingerprint.webgl,
                "canvas": session.fingerprint.canvas,
                "audio": session.fingerprint.audio,
                "viewport": list(session.fingerprint.viewport),
                "timezone": session.fingerprint.timezone,
                "language": session.fingerprint.language,
                "color_depth": session.fingerprint.color_depth,
                "device_memory": session.fingerprint.device_memory,
                "hardware_concurrency": session.fingerprint.hardware_concurrency,
                "plugins": session.fingerprint.plugins,
                "fonts": session.fingerprint.fonts,
            },
            "cookies": session.cookies,
            "local_storage": session.local_storage,
            "session_storage": session.session_storage,
            "created_at": session.created_at.isoformat(),
            "last_used": session.last_used.isoformat(),
            "request_count": session.request_count,
            "is_valid": session.is_valid,
        }

        async with aiofiles.open(session_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))

    async def _load_existing_sessions(self) -> None:
        """加载现有会话"""
        for session_file in self._storage_dir.glob("sess_*.json"):
            try:
                async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)

                session = SessionData(
                    session_id=data["session_id"],
                    user_agent=UserAgentData(
                        user_agent=data["user_agent"]["user_agent"],
                        browser=BrowserType(data["user_agent"]["browser"]),
                        browser_version=data["user_agent"]["browser_version"],
                        platform=PlatformType(data["user_agent"]["platform"]),
                        platform_version=data["user_agent"]["platform_version"],
                        engine=data["user_agent"]["engine"],
                        engine_version=data["user_agent"]["engine_version"],
                    ),
                    fingerprint=BrowserFingerprint(
                        webgl=data["fingerprint"]["webgl"],
                        canvas=data["fingerprint"]["canvas"],
                        audio=data["fingerprint"]["audio"],
                        viewport=tuple(data["fingerprint"]["viewport"]),
                        timezone=data["fingerprint"]["timezone"],
                        language=data["fingerprint"]["language"],
                        color_depth=data["fingerprint"]["color_depth"],
                        device_memory=data["fingerprint"]["device_memory"],
                        hardware_concurrency=data["fingerprint"]["hardware_concurrency"],
                        plugins=data["fingerprint"]["plugins"],
                        fonts=data["fingerprint"]["fonts"],
                    ),
                    cookies=data["cookies"],
                    local_storage=data["local_storage"],
                    session_storage=data["session_storage"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_used=datetime.fromisoformat(data["last_used"]),
                    request_count=data["request_count"],
                    is_valid=data["is_valid"],
                )

                if await self.check_session_valid(session):
                    self._sessions[session.session_id] = session

            except Exception:
                continue

    async def _cleanup_old_sessions(self) -> None:
        """清理旧会话"""
        sessions_to_remove = []

        for session_id, session in self._sessions.items():
            if not await self.check_session_valid(session):
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            await self.delete_session(session_id)

        if len(self._sessions) >= self._max_sessions:
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].last_used
            )

            to_remove = len(self._sessions) - self._max_sessions + 10
            for session_id, _ in sorted_sessions[:to_remove]:
                await self.delete_session(session_id)
