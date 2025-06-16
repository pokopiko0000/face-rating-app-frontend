import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'flagcdn.com',
      },
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8003',
        pathname: '/**',
      },
    ],
  },
  reactStrictMode: true,
};

export default nextConfig;
