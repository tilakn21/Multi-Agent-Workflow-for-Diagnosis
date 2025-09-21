import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  experimental: {},
  allowedDevOrigins: [
    "https://6000-firebase-studio-1758312907097.cluster-y75up3teuvc62qmnwys4deqv6y.cloudworkstations.dev",
  ],
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "placehold.co",
        port: "",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "images.unsplash.com",
        port: "",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "picsum.photos",
        port: "",
        pathname: "/**",
      },
    ],
  },
  webpack(config) {
    config.externals = [
      ...(config.externals || []),
      "@opentelemetry/sdk-node",
      "@genkit-ai/firebase",
    ];
    config.resolve.alias = {
      ...config.resolve.alias,
      "handlebars/runtime": "handlebars/dist/cjs/handlebars.runtime",
      handlebars: "handlebars/dist/cjs/handlebars",
    };

    return config;
  },
};

export default nextConfig;
