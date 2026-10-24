import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Do not generate AGENTS.md / agent rule files during `next dev`.
  agentRules: false,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
