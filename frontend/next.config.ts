import { NextConfig } from 'next'

/**
 * @fileoverview Next.js Configuration - Production-ready settings for semantic search app
 * @learning Demonstrates Next.js optimization and external resource configuration
 */

const config: NextConfig = {
  /**
   * Learning: Use src directory for better organization
   * This tells Next.js to look for app directory inside src/
   */
  experimental: {
    // Enable optimized bundling for large applications
    optimizePackageImports: ['lucide-react'],
  },

  /**
   * Learning: Image optimization configuration
   * Next.js Image component requires explicit domains for external images
   * This ensures security and enables optimization features
   */
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'picsum.photos',
        port: '',
        pathname: '/**',
      },
      // Add other image providers as needed
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'http',
        hostname: 'assets.myntassets.com',
        port: '',
        pathname: '/**',
      }
    ],
    // Enable modern image formats for better performance
    formats: ['image/webp', 'image/avif'],
  },

  /**
   * Learning: Performance optimizations for production
   */
  compiler: {
    // Remove console.logs in production builds
    removeConsole: process.env.NODE_ENV === 'production',
  },
}

export default config
