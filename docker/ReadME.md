# Docker for lighthouse and browsertime

## Dependencies

Install Docker. https://docs.docker.com/engine/install/ubuntu/

## How to build Docker?
Open the terminal inside docker directory and run the following command:
```
	docker build . --tag lighthouse-test
```

## How to run lighthouse?
```
	mkdir data # for storing results
	pwd # This command will print the absolute path
	docker run -v <absolute-path>:/data lighthouse-test lighthouse --output json --disable-device-emulation --throttling.cpuSlowdownMultiplier=1 --output-path=/data/<filename.json> <site>
```

## How to run browsertime?
```
	mkdir data # for storing results
	pwd # This command will print the absolute path
	docker run -v <absolute-path>/data:/data lighthouse-test browsertime --pageLoadStrategy normal --resultDir data --output <filename> <site>
```

These commands will save result files in the data folder. We can add extra fields to both of these commands. 

Here are the list of options for lighthouse:
```
lighthouse <url> <options>

Logging:
      --verbose  Displays verbose logging  [boolean] [default: false]
      --quiet    Displays no progress, debug logs, or errors  [boolean] [default: false]

Configuration:
      --save-assets                  Save the trace contents & devtools logs to disk  [boolean] [default: false]
      --list-all-audits              Prints a list of all available audits and exits  [boolean] [default: false]
      --list-locales                 Prints a list of all supported locales and exits  [boolean] [default: false]
      --list-trace-categories        Prints a list of all required trace categories and exits  [boolean] [default: false]
      --print-config                 Print the normalized config for the given config and options, then exit.  [boolean] [default: false]
      --additional-trace-categories  Additional categories to capture with the trace (comma-delimited).  [string]
      --config-path                  The path to the config JSON.
                                                 An example config file: lighthouse-core/config/lr-desktop-config.js  [string]
      --preset                       Use a built-in configuration.
                                               WARNING: If the --config-path flag is provided, this preset will be ignored.  [string] [choices: "perf", "experimental", "desktop"]
      --chrome-flags                 Custom flags to pass to Chrome (space-delimited). For a full list of flags, see https://bit.ly/chrome-flags
                                                 Additionally, use the CHROME_PATH environment variable to use a specific Chrome binary. Requires Chromium version 66.0 or later. If omitted, any detected Chrome Canary or Chrome stable will be used.  [string] [default: ""]
      --port                         The port to use for the debugging protocol. Use 0 for a random port  [number] [default: 0]
      --hostname                     The hostname to use for the debugging protocol.  [string] [default: "127.0.0.1"]
      --form-factor                  Determines how performance metrics are scored and if mobile-only audits are skipped. For desktop, --preset=desktop instead.  [string] [choices: "mobile", "desktop"]
      --screenEmulation              Sets screen emulation parameters. See also --preset. Use --screenEmulation.disabled to disable. Otherwise set these 4 parameters individually: --screenEmulation.mobile --screenEmulation.width=360 --screenEmulation.height=640 --screenEmulation.deviceScaleFactor=2
      --emulatedUserAgent            Sets useragent emulation  [string]
      --max-wait-for-load            The timeout (in milliseconds) to wait before the page is considered done loading and the run should continue. WARNING: Very high values can lead to large traces and instability  [number]
      --enable-error-reporting       Enables error reporting, overriding any saved preference. --no-enable-error-reporting will do the opposite. More: https://git.io/vFFTO  [boolean]
  -G, --gather-mode                  Collect artifacts from a connected browser and save to disk. (Artifacts folder path may optionally be provided). If audit-mode is not also enabled, the run will quit early.
  -A, --audit-mode                   Process saved artifacts from disk. (Artifacts folder path may be provided, otherwise defaults to ./latest-run/)
      --only-audits                  Only run the specified audits  [array]
      --only-categories              Only run the specified categories. Available categories: accessibility, best-practices, performance, pwa, seo  [array]
      --skip-audits                  Run everything except these audits  [array]
      --budget-path                  The path to the budget.json file for LightWallet.  [string]

Output:
      --output       Reporter for the results, supports multiple values. choices: "json", "html", "csv"  [array] [default: ["html"]]
      --output-path  The file path to output the results. Use 'stdout' to write to stdout.
                       If using JSON output, default is stdout.
                       If using HTML or CSV output, default is a file in the working directory with a name based on the test URL and date.
                       If using multiple outputs, --output-path is appended with the standard extension for each output type. "reports/my-run" -> "reports/my-run.report.html", "reports/my-run.report.json", etc.
                       Example: --output-path=./lighthouse-results.html  [string]
      --view         Open HTML report in your browser  [boolean] [default: false]

Options:
      --help                               Show help  [boolean]
      --version                            Show version number  [boolean]
      --cli-flags-path                     The path to a JSON file that contains the desired CLI flags to apply. Flags specified at the command line will still override the file-based ones.
      --debug-navigation                   Pause after page load to wait for permission to continue the run, evaluate `continueLighthouseRun` in the console to continue.  [boolean]
      --fraggle-rock                       [EXPERIMENTAL] Use the new Fraggle Rock navigation runner to gather results.  [boolean] [default: false]
      --locale                             The locale/language the report should be formatted in
      --blocked-url-patterns               Block any network requests to the specified URL patterns  [array]
      --disable-storage-reset              Disable clearing the browser cache and other storage APIs before a run  [boolean]
      --throttling-method                  Controls throttling method  [string] [choices: "devtools", "provided", "simulate"]
      --throttling
      --throttling.rttMs                   Controls simulated network RTT (TCP layer)
      --throttling.throughputKbps          Controls simulated network download throughput
      --throttling.requestLatencyMs        Controls emulated network RTT (HTTP layer)
      --throttling.downloadThroughputKbps  Controls emulated network download throughput
      --throttling.uploadThroughputKbps    Controls emulated network upload throughput
      --throttling.cpuSlowdownMultiplier   Controls simulated + emulated CPU throttling
      --extra-headers                      Set extra HTTP Headers to pass with request
      --precomputed-lantern-data-path      Path to the file where lantern simulation data should be read from, overwriting the lantern observed estimates for RTT and server latency.  [string]
      --lantern-data-output-path           Path to the file where lantern simulation data should be written to, can be used in a future run with the `precomputed-lantern-data-path` flag.  [string]
      --plugins                            Run the specified plugins  [array]
      --channel  [string] [default: "cli"]
      --chrome-ignore-default-flags  [boolean] [default: false]

Examples:
  lighthouse <url> --view                                                                            Opens the HTML report in a browser after the run completes
  lighthouse <url> --config-path=./myconfig.js                                                       Runs Lighthouse with your own configuration: custom audits, report generation, etc.
  lighthouse <url> --output=json --output-path=./report.json --save-assets                           Save trace, screenshots, and named JSON report.
  lighthouse <url> --screenEmulation.disabled --throttling-method=provided --no-emulated-user-agent  Disable emulation and all throttling
  lighthouse <url> --chrome-flags="--window-size=412,660"                                            Launch Chrome with a specific window size
  lighthouse <url> --quiet --chrome-flags="--headless"                                               Launch Headless Chrome, turn off logging
  lighthouse <url> --extra-headers "{\"Cookie\":\"monster=blue\", \"x-men\":\"wolverine\"}"          Stringify'd JSON HTTP Header key/value pairs to send in requests
  lighthouse <url> --extra-headers=./path/to/file.json                                               Path to JSON file of HTTP Header key/value pairs to send in requests
  lighthouse <url> --only-categories=performance,pwa                                                 Only run the specified categories. Available categories: accessibility, best-practices, performance, pwa, seo
```

Here are the options for browsertime:
```
browsertime [options] <url>/<scriptFile>

timeouts
      --timeouts.browserStart       Timeout when waiting for browser to start, in milliseconds  [number] [default: 60000]
      --timeouts.pageLoad           Timeout when waiting for url to load, in milliseconds  [number] [default: 300000]
      --timeouts.script             Timeout when running browser scripts, in milliseconds  [number] [default: 120000]
      --timeouts.pageCompleteCheck  Timeout when waiting for page to complete loading, in milliseconds  [number] [default: 300000]

chrome
      --chrome.args                                              Extra command line arguments to pass to the Chrome process (e.g. --no-sandbox). To add multiple arguments to Chrome, repeat --chrome.args once per argument.
      --chrome.binaryPath                                        Path to custom Chrome binary (e.g. Chrome Canary). On OS X, the path should be to the binary inside the app bundle, e.g. "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"
      --chrome.chromedriverPath                                  Path to custom ChromeDriver binary. Make sure to use a ChromeDriver version that's compatible with the version of Chrome you're using
      --chrome.chromedriverPort                                  Specify "--port" args for chromedriver prcocess  [number]
      --chrome.mobileEmulation.deviceName                        Name of device to emulate. Works only standalone (see list in Chrome DevTools, but add phone like 'iPhone 6'). This will override your userAgent string.
      --chrome.mobileEmulation.width                             Width in pixels of emulated mobile screen (e.g. 360)  [number]
      --chrome.mobileEmulation.height                            Height in pixels of emulated mobile screen (e.g. 640)  [number]
      --chrome.mobileEmulation.pixelRatio                        Pixel ratio of emulated mobile screen (e.g. 2.0)
      --chrome.android.package                                   Run Chrome on your Android device. Set to com.android.chrome for default Chrome version. You need to have adb installed to make this work.
      --chrome.android.activity                                  Name of the Activity hosting the WebView.
      --chrome.android.process                                   Process name of the Activity hosting the WebView. If not given, the process name is assumed to be the same as chrome.android.package.
      --chrome.android.deviceSerial                              Choose which device to use. If you do not set it, first device will be used.
      --chrome.traceCategories                                   A comma separated list of Tracing event categories to include in the Trace log. Default no trace categories is collected.  [string]
      --chrome.traceCategory                                     Add a trace category to the default ones. Use --chrome.traceCategory multiple times if you want to add multiple categories. Example: --chrome.traceCategory disabled-by-default-v8.cpu_profiler  [string]
      --chrome.enableTraceScreenshots, --enableTraceScreenshots  Include screenshots in the trace log (enabling the trace category disabled-by-default-devtools.screenshot).  [boolean]
      --chrome.enableChromeDriverLog                             Log Chromedriver communication to a log file.  [boolean]
      --chrome.enableVerboseChromeDriverLog                      Log verboose Chromedriver communication to a log file.  [boolean]
      --chrome.timeline, --chrome.trace                          Collect the timeline data. Drag and drop the JSON in your Chrome detvools timeline panel or check out the CPU metrics in the Browsertime.json  [boolean]
      --chrome.collectPerfLog                                    Collect performance log from Chrome with Page and Network events and save to disk.  [boolean]
      --chrome.collectNetLog                                     Collect network log from Chrome and save to disk.  [boolean]
      --chrome.netLogCaptureMode                                 Choose capture mode for Chromes netlog.  [choices: "Default", "IncludeSensitive", "Everything"] [default: "IncludeSensitive"]
      --chrome.collectConsoleLog                                 Collect Chromes console log and save to disk.  [boolean]
      --chrome.appendToUserAgent                                 Append to the user agent.  [string]
      --chrome.noDefaultOptions                                  Prevent Browsertime from setting its default options for Chrome  [boolean]
      --chrome.CPUThrottlingRate                                 Enables CPU throttling to emulate slow CPUs. Throttling rate as a slowdown factor (1 is no throttle, 2 is 2x slowdown, etc)  [number]
      --chrome.includeResponseBodies                             Include response bodies in the HAR file.  [choices: "none", "all", "html"] [default: "none"]
      --chrome.cdp.performance                                   Collect Chrome perfromance metrics from Chrome DevTools Protocol  [boolean] [default: true]
      --chrome.blockDomainsExcept, --blockDomainsExcept          Block all domains except this domain. Use it multiple time to keep multiple domains. You can also wildcard domains like *.sitespeed.io. Use this when you wanna block out all third parties.
      --chrome.ignoreCertificateErrors                           Make Chrome ignore certificate errors.  Defaults to true.  [boolean] [default: true]

firefox
      --firefox.binaryPath                      Path to custom Firefox binary (e.g. Firefox Nightly). On OS X, the path should be to the binary inside the app bundle, e.g. /Applications/Firefox.app/Contents/MacOS/firefox-bin
      --firefox.geckodriverPath                 Path to custom geckodriver binary. Make sure to use a geckodriver version that's compatible with the version of Firefox (Gecko) you're using
      --firefox.geckodriverArgs                 Flags passed in to Geckodriver see https://firefox-source-docs.mozilla.org/testing/geckodriver/Flags.html. Use it like --firefox.geckodriverArgs="--marionette-port"  --firefox.geckodriverArgs=1027  [string]
      --firefox.appendToUserAgent               Append to the user agent.  [string]
      --firefox.nightly                         Use Firefox Nightly. Works on OS X. For Linux you need to set the binary path.  [boolean]
      --firefox.beta                            Use Firefox Beta. Works on OS X. For Linux you need to set the binary path.  [boolean]
      --firefox.developer                       Use Firefox Developer. Works on OS X. For Linux you need to set the binary path.  [boolean]
      --firefox.preference                      Extra command line arguments to pass Firefox preferences by the format key:value To add multiple preferences, repeat --firefox.preference once per argument.
      --firefox.args                            Extra command line arguments to pass to the Firefox process (e.g. --MOZ_LOG). To add multiple arguments to Firefox, repeat --firefox.args once per argument.
      --firefox.includeResponseBodies           Include response bodies in HAR  [choices: "none", "all", "html"] [default: "none"]
      --firefox.appconstants                    Include Firefox AppConstants information in the results  [boolean] [default: false]
      --firefox.acceptInsecureCerts             Accept insecure certs  [boolean]
      --firefox.windowRecorder                  Use the internal compositor-based Firefox window recorder to emit PNG files for each frame that is a meaningful change.  The PNG output will further be merged into a variable frame rate video for analysis. Use this instead of ffmpeg to record a video (you still need the --video flag).  [boolean] [default: false]
      --firefox.memoryReport                    Measure firefox resident memory after each iteration.  [boolean] [default: false]
      --firefox.memoryReportParams.minizeFirst  Force a collection before dumping and measuring the memory report.  [boolean] [default: false]
      --firefox.geckoProfiler                   Collect a profile using the internal gecko profiler  [boolean] [default: false]
      --firefox.geckoProfilerParams.features    Enabled features during gecko profiling  [string] [default: "js,stackwalk,leaf"]
      --firefox.geckoProfilerParams.threads     Threads to profile.  [string] [default: "GeckoMain,Compositor,Renderer"]
      --firefox.geckoProfilerParams.interval    Sampling interval in ms.  Defaults to 1 on desktop, and 4 on android.  [number]
      --firefox.geckoProfilerParams.bufferSize  Buffer size in elements. Default is ~90MB.  [number] [default: 13107200]
      --firefox.perfStats                       Collect gecko performance statistics as measured internally by the firefox browser. See https://searchfox.org/mozilla-central/source/tools/performance/PerfStats.h#24-33  [boolean] [default: false]
      --firefox.perfStatsParams.mask            Mask to decide which features to enable  [number] [default: 4294967295]
      --firefox.collectMozLog                   Collect the MOZ HTTP log (by default). See --firefox.setMozLog if you need to specify the logs you wish to gather.  [boolean]
      --firefox.setMozLog                       Use in conjunction with firefox.collectMozLog to set MOZ_LOG to something specific. Without this, the HTTP logs will be collected by default  [default: "timestamp,nsHttp:5,cache2:5,nsSocketTransport:5,nsHostResolver:5"]
      --firefox.disableBrowsertimeExtension     Disable installing the browsertime extension.  [boolean]
      --firefox.noDefaultPrefs                  Prevents browsertime from setting its default preferences.  [boolean] [default: false]
      --firefox.disableSafeBrowsing             Disable safebrowsing.  [boolean] [default: true]
      --firefox.disableTrackingProtection       Disable Tracking Protection.  [boolean] [default: true]
      --firefox.android.package                 Run Firefox or a GeckoView-consuming App on your Android device. Set to org.mozilla.geckoview_example for default Firefox version. You need to have adb installed to make this work.
      --firefox.android.activity                Name of the Activity hosting the GeckoView.
      --firefox.android.deviceSerial            Choose which device to use. If you do not set it, first device will be used.
      --firefox.android.intentArgument          Configure how the Android intent is launched.  Passed through to `adb shell am start ...`; follow the format at https://developer.android.com/studio/command-line/adb#IntentSpec. To add multiple arguments, repeat --firefox.android.intentArgument once per argument.
      --firefox.profileTemplate                 Profile template directory that will be cloned and used as the base of each profile each instance of Firefox is launched against.  Use this to pre-populate databases with certificates, tracking protection lists, etc.

selenium
      --selenium.url  URL to a running Selenium server (e.g. to run a browser on another machine).

video
      --videoParams.framerate          Frames per second  [default: 30]
      --videoParams.crf                Constant rate factor see https://trac.ffmpeg.org/wiki/Encode/H.264#crf  [default: 23]
      --videoParams.addTimer           Add timer and metrics to the video.  [boolean] [default: true]
      --videoParams.debug              Turn on debug to record a video with all pre/post and scripts/URLS you test in one iteration. Visual Metrics will then automatically be disabled.  [boolean] [default: false]
      --videoParams.keepOriginalVideo  Keep the original video. Use it when you have a Visual Metrics bug and want to create an issue at GitHub  [boolean] [default: false]
      --videoParams.filmstripFullSize  Keep original sized screenshots. Will make the run take longer time  [boolean] [default: false]
      --videoParams.filmstripQuality   The quality of the filmstrip screenshots. 0-100.  [default: 75]
      --videoParams.createFilmstrip    Create filmstrip screenshots.  [boolean] [default: true]
      --videoParams.nice               Use nice when running FFMPEG during the run. A value from -20 to 19  https://linux.die.net/man/1/nice  [default: 0]
      --videoParams.convert            Convert the original video to a viewable format (for most video players). Turn that off to make a faster run.  [boolean] [default: true]
      --videoParams.threads            Number of threads to use for video recording. Default is determined by ffmpeg.  [default: 0]

edge
      --edge.edgedriverPath  Path to custom msedgedriver version (need to match your Egde version).
      --edge.binaryPath      Path to custom Edge binary

safari
      --safari.ios                   Use Safari on iOS. You need to choose browser Safari and iOS to run on iOS.  [boolean] [default: false]
      --safari.deviceName            Set the device name. Device names for connected devices are shown in iTunes.
      --safari.deviceUDID            Set the device UDID. If Xcode is installed, UDIDs for connected devices are available via the output of "xcrun simctl list devices" and in the Device and Simulators window (accessed in Xcode via "Window > Devices and Simulators")
      --safari.deviceType            Set the device type. If the value of safari:deviceType is `iPhone`, safaridriver will only create a session using an iPhone device or iPhone simulator. If the value of safari:deviceType is `iPad`, safaridriver will only create a session using an iPad device or iPad simulator.
      --safari.useTechnologyPreview  Use Safari Technology Preview  [boolean] [default: false]
      --safari.diagnose              When filing a bug report against safaridriver, it is highly recommended that you capture and include diagnostics generated by safaridriver. Diagnostic files are saved to ~/Library/Logs/com.apple.WebDriver/
      --safari.useSimulator          If the value of useSimulator is true, safaridriver will only use iOS Simulator hosts. If the value of safari:useSimulator is false, safaridriver will not use iOS Simulator hosts. NOTE: An Xcode installation is required in order to run WebDriver tests on iOS Simulator hosts.  [boolean] [default: false]

Screenshot
      --screenshot                             Save one screenshot per iteration.  [boolean] [default: false]
      --screenshotLCP                          Save one screenshot per iteration that shows the largest contentful paint element (if the browser supports LCP).  [boolean] [default: false]
      --screenshotLS                           Save one screenshot per iteration that shows the layout shift elements (if the browser supports layout shift).  [boolean] [default: false]
      --screenshotParams.type                  Set the file type of the screenshot  [choices: "png", "jpg"] [default: "jpg"]
      --screenshotParams.png.compressionLevel  zlib compression level  [default: 6]
      --screenshotParams.jpg.quality           Quality of the JPEG screenshot. 1-100  [default: 80]
      --screenshotParams.maxSize               The max size of the screenshot (width and height).  [default: 2000]

proxy
      --proxy.pac     Proxy auto-configuration (URL)  [string]
      --proxy.ftp     Ftp proxy (host:port)  [string]
      --proxy.http    Http proxy (host:port)  [string]
      --proxy.https   Https proxy (host:port)  [string]
      --proxy.bypass  Comma separated list of hosts to connect to directly, bypassing other proxies for that host  [string]

connectivity
  -c, --connectivity.profile                              The connectivity profile.  [choices: "4g", "3g", "3gfast", "3gslow", "3gem", "2g", "cable", "native", "custom"] [default: "native"]
      --connectivity.down, --connectivity.downstreamKbps  This option requires --connectivity.profile be set to "custom".
      --connectivity.up, --connectivity.upstreamKbps      This option requires --connectivity.profile be set to "custom".
      --connectivity.rtt, --connectivity.latency          This option requires --connectivity.profile be set to "custom".
      --connectivity.variance                             This option requires --connectivity.engine be set to "throttle". It will add a variance to the rtt between each run. --connectivity.variance 2 means it will run with a random variance of max 2% between runs.
      --connectivity.alias                                Give your connectivity profile a custom name
      --connectivity.engine                               The engine for connectivity. Throttle works on Mac and tc based Linux. For mobile you can use Humble if you have a Humble setup. Use external if you set the connectivity outside of Browsertime. The best way do to this is described in https://github.com/sitespeedio/browsertime#connectivity.  [choices: "external", "throttle", "humble"] [default: "external"]
      --connectivity.throttle.localhost                   Add latency/delay on localhost. Perfect for testing with WebPageReplay  [boolean] [default: false]
      --connectivity.humble.url                           The path to your Humble instance. For example http://raspberrypi:3000  [string]

Options:
      --cpu                                         Easy way to enable both chrome.timeline for Chrome and geckoProfile for Firefox  [boolean]
      --androidPower                                Enables android power testing - charging must be disabled for this.(You have to disable charging yourself for this - it depends on the phone model).  [boolean]
      --video                                       Record a video and store the video. Set it to false to remove the video that is created by turning on visualMetrics. To remove fully turn off video recordings, make sure to set video and visualMetrics to false. Requires FFMpeg to be installed.  [boolean]
      --visualMetrics                               Collect Visual Metrics like First Visual Change, SpeedIndex, Perceptual Speed Index and Last Visual Change. Requires FFMpeg and Python dependencies  [boolean]
      --visualElements, --visuaElements             Collect Visual Metrics from elements. Works only with --visualMetrics turned on. By default you will get visual metrics from the largest image within the view port and the largest h1. You can also configure to pickup your own defined elements with --scriptInput.visualElements  [boolean]
      --visualMetricsPerceptual                     Collect Perceptual Speed Index when you run --visualMetrics.  [boolean]
      --visualMetricsContentful                     Collect Contentful Speed Index when you run --visualMetrics.  [boolean]
      --scriptInput.visualElements                  Include specific elements in visual elements. Give the element a name and select it with document.body.querySelector. Use like this: --scriptInput.visualElements name:domSelector see https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Selectors. Add multiple instances to measure multiple elements. Visual Metrics will use these elements and calculate when they are visible and fully rendered.
      --scriptInput.longTask, --minLongTaskLength   Set the minimum length of a task to be categorised as a CPU Long Task. It can never be smaller than 50. The value is in ms and only works in Chromium browsers at the moment.  [number] [default: 50]
  -b, --browser                                     Specify browser. Safari only works on OS X/iOS. Edge only work on OS that supports Edge.  [choices: "chrome", "firefox", "edge", "safari"] [default: "chrome"]
      --android                                     Short key to use Android. Defaults to use com.android.chrome unless --browser is specified.  [boolean] [default: false]
      --androidRooted                               If your phone is rooted you can use this to set it up following Mozillas best practice for stable metrics.  [boolean] [default: false]
      --androidBatteryTemperatureLimit              Do the battery temperature need to be below a specific limit before we start the test?
      --androidBatteryTemperatureWaitTimeInSeconds  How long time to wait (in seconds) if the androidBatteryTemperatureWaitTimeInSeconds is not met before the next try  [default: 120]
      --androidBatteryTemperatureReboot             If your phone does not get the minimum temperature aftet the wait time, reboot the phone.  [boolean] [default: false]
      --androidPretestPowerPress                    Press the power button on the phone before a test starts.  [boolean] [default: false]
      --androidVerifyNetwork                        Before a test start, verify that the device has a Internet connection by pinging 8.8.8.8 (or a configurable domain with --androidPingAddress)  [boolean] [default: false]
      --processStartTime                            Capture browser process start time (in milliseconds). Android only for now.  [boolean] [default: false]
      --pageCompleteCheck                           Supply a JavaScript (inline or JavaScript file) that decides when the browser is finished loading the page and can start to collect metrics. The JavaScript snippet is repeatedly queried to see if page has completed loading (indicated by the script returning true). Use it to fetch timings happening after the loadEventEnd. By default the tests ends 2 seconds after loadEventEnd. Also checkout --pageCompleteCheckInactivity and --pageCompleteCheckPollTimeout
      --pageCompleteWaitTime                        How long time you want to wait for your pageComplteteCheck to finish, after it is signaled to closed. Extra parameter passed on to your pageCompleteCheck.  [default: 8000]
      --pageCompleteCheckInactivity                 Alternative way to choose when to end your test. This will wait for 2 seconds of inactivity that happens after loadEventEnd.  [boolean] [default: false]
      --pageCompleteCheckPollTimeout                The time in ms to wait for running the page complete check the next time.  [number] [default: 1500]
      --pageCompleteCheckStartWait                  The time in ms to wait for running the page complete check for the first time. Use this when you have a pageLoadStrategy set to none  [number] [default: 5000]
      --pageLoadStrategy                            Set the strategy to waiting for document readiness after a navigation event. After the strategy is ready, your pageCompleteCheck will start runninhg.  [string] [choices: "eager", "none", "normal"] [default: "none"]
  -n, --iterations                                  Number of times to test the url (restarting the browser between each test)  [number] [default: 3]
      --prettyPrint                                 Enable to print json/har with spaces and indentation. Larger files, but easier on the eye.  [boolean] [default: false]
      --delay                                       Delay between runs, in milliseconds  [number] [default: 0]
      --timeToSettle                                Extra time added for the browser to settle before starting to test a URL. This delay happens after the browser was opened and before the navigation to the URL  [number] [default: 0]
      --webdriverPageload                           Use webdriver.get to initialize the page load instead of window.location.  [boolean] [default: false]
  -r, --requestheader                               Request header that will be added to the request. Add multiple instances to add multiple request headers. Works for Firefox and Chrome. Use the following format key:value
      --cookie                                      Cookie that will be added to the request. Add multiple instances to add multiple request cookies. Works for Firefox and Chrome. Use the following format cookieName=cookieValue
      --injectJs                                    Inject JavaScript into the current page at document_start. Works for Firefox and Chrome. More info: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/contentScripts
      --block                                       Domain to block. Add multiple instances to add multiple domains that will be blocked. If you use Chrome you can also use --blockDomainsExcept (that is more performant). Works for Firefox and Chrome.
      --percentiles                                 The percentile values within the data browsertime will calculate and report. This argument uses Yargs arrays and you you to set them correctly it is recommended to use a configuraration file instead.  [array] [default: [0,10,90,99,100]]
      --decimals                                    The decimal points browsertime statistics round to.  [number] [default: 0]
      --iqr                                         Use IQR, or Inter Quartile Range filtering filters data based on the spread of the data. See  https://en.wikipedia.org/wiki/Interquartile_range. In some cases, IQR filtering may not filter out anything. This can happen if the acceptable range is wider than the bounds of your dataset.  [boolean] [default: false]
      --cacheClearRaw                               Use internal browser functionality to clear browser cache between runs instead of only using Selenium.  [boolean] [default: false]
      --basicAuth                                   Use it if your server is behind Basic Auth. Format: username@password (Only Chrome and Firefox at the moment).
      --preScript, --setUp                          Selenium script(s) to run before you test your URL/script. They will run outside of the analyse phase. Note that --preScript can be passed multiple times.
      --postScript, --tearDown                      Selenium script(s) to run after you test your URL. They will run outside of the analyse phase. Note that --postScript can be passed multiple times.
      --script                                      Add custom Javascript to run after the page has finished loading to collect metrics. If a single js file is specified, it will be included in the category named "custom" in the output json. Pass a folder to include all .js scripts in the folder, and have the folder name be the category. Note that --script can be passed multiple times.
      --userAgent                                   Override user agent
      --appendToUserAgent                           Append a String to the user agent. Works in Chrome/Edge and Firefox.
  -q, --silent                                      Only output info in the logs, not to the console. Enter twice to suppress summary line.  [count]
  -o, --output                                      Specify file name for Browsertime data (ex: 'browsertime'). Unless specified, file will be named browsertime.json
      --har                                         Specify file name for .har file (ex: 'browsertime'). Unless specified, file will be named browsertime.har
      --skipHar                                     Pass --skipHar to not collect a HAR file.  [boolean]
      --gzipHar                                     Pass --gzipHar to gzip the HAR file  [boolean]
      --config                                      Path to JSON config file. You can also use a .browsertime.json file that will automatically be found by Browsertime using find-up.
      --viewPort                                    Size of browser window WIDTHxHEIGHT or "maximize". Note that "maximize" is ignored for xvfb.
      --resultDir                                   Set result directory for the files produced by Browsertime
      --useSameDir                                  Store all files in the same structure and do not use the path structure released in 4.0. Use this only if you are testing ONE URL.
      --xvfb                                        Start xvfb before the browser is started  [boolean] [default: false]
      --xvfbParams.display                          The display used for xvfb  [default: 99]
      --tcpdump                                     Collect a tcpdump for each tested URL.  [boolean] [default: false]
      --tcpdumpPacketBuffered                       Use together with --tcpdump to save each packet directly to the file, instead of buffering.  [boolean] [default: false]
      --urlAlias                                    Use an alias for the URL. You need to pass on the same amount of alias as URLs. The alias is used as the name of the URL and used for filepath. Pass on multiple --urlAlias for multiple alias/URLs. You can also add alias direct in your script.  [string]
      --preURL                                      A URL that will be accessed first by the browser before the URL that you wanna analyze. Use it to fill the browser cache.
      --preURLDelay                                 Delay between preURL and the URL you want to test (in milliseconds)  [default: 1500]
      --userTimingWhitelist                         All userTimings are captured by default this option takes a regex that will whitelist which userTimings to capture in the results.
      --headless                                    Run the browser in headless mode. Works for Firefox and Chrome.  [boolean] [default: false]
      --gnirehtet                                   Start gnirehtet and reverse tethering the traffic from your Android phone.  [boolean] [default: false]
      --extension                                   Path to a WebExtension to be installed in the browser. Note that --extension can be passed multiple times.
      --spa                                         Convenient parameter to use if you test a SPA application: will automatically wait for X seconds after last network activity and use hash in file names. Read more: https://www.sitespeed.io/documentation/sitespeed.io/spa/  [boolean] [default: false]
      --browserRestartTries                         If the browser fails to start, you can retry to start it this amount of times.  [number] [default: 3]
      --preWarmServer                               Do pre test requests to the URL(s) that you want to test that is not measured. Do that to make sure your web server is ready to serve. The pre test requests is done with another browser instance that is closed after pre testing is done.  [boolean] [default: false]
      --preWarmServerWaitTime                       The wait time before you start the real testing after your pre-cache request.  [number] [default: 5000]
  -h, --help                                        Show help  [boolean]
  -V, --version                                     Show version number  [boolean]
```