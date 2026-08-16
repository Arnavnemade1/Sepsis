import { Config } from "@remotion/cli/config";

/**
 * Remotion CLI render settings. Local only — this never runs on the host that
 * builds the website.
 *
 * No overrideBundlerConfig / @remotion/tailwind-v4 needed: the composition uses
 * inline styles fed from lib/tokens.ts rather than Tailwind classes, which is
 * what makes the <Player> output and the rendered MP4 identical for free.
 */
Config.setVideoImageFormat("jpeg");
Config.setCodec("h264");
Config.setCrf(18);
Config.setConcurrency(4);
