/**
 * @fileoverview Main Semantic Image Search Page - Fashion AI Search Interface
 * @learning Demonstrates full-stack semantic search application with Next.js 14
 * @concepts App Router, Client Components, State Management, Responsive Design
 */

import React from 'react'
import { SearchInterface } from '@/components/image-search/search-interface'

/**
 * Learning: Next.js 14 App Router with Server Components by default
 * This page is a Server Component - it runs on the server and sends HTML to client
 * Benefits: Better SEO, faster initial page load, reduced JavaScript bundle
 */
export default function HomePage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header Section */}
      <div className="bg-white border-b">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl lg:text-5xl font-bold text-gray-900">
              AI-Powered Fashion Search
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Discover fashion items using natural language descriptions or upload images 
              to find visually similar products with advanced semantic search technology
            </p>
            
            {/* Learning Badges */}
            <div className="flex flex-wrap justify-center gap-2 mt-6">
              <span className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full">
                🤖 Vector Embeddings
              </span>
              <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">
                🔍 Semantic Search
              </span>
              <span className="px-3 py-1 bg-purple-100 text-purple-800 text-sm rounded-full">
                📊 Similarity Scoring
              </span>
              <span className="px-3 py-1 bg-orange-100 text-orange-800 text-sm rounded-full">
                🎯 CLIP Technology
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Search Interface */}
      <SearchInterface />

      {/* Footer with Learning Resources */}
      <footer className="bg-white border-t mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Learn About Semantic Search
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              <div className="space-y-2">
                <h4 className="font-medium text-gray-900">Vector Embeddings</h4>
                <p className="text-sm text-gray-600">
                  Neural networks convert images and text into high-dimensional vectors 
                  that capture semantic meaning for similarity comparison.
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-gray-900">CLIP Models</h4>
                <p className="text-sm text-gray-600">
                  Contrastive Language-Image Pre-training enables cross-modal search 
                  between text descriptions and visual content.
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-gray-900">Cosine Similarity</h4>
                <p className="text-sm text-gray-600">
                  Mathematical measure of similarity between vectors, perfect for 
                  comparing normalized neural network embeddings.
                </p>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </main>
  )
}

/**
 * Learning: Metadata for SEO and social sharing
 * Next.js 14 App Router uses this export for page metadata
 */
export const metadata = {
  title: 'AI Fashion Search - Semantic Image Search Demo',
  description: 'Discover fashion items using AI-powered semantic search with text descriptions or image uploads. Built with Next.js, MongoDB Vector Search, and CLIP embeddings.',
  keywords: 'AI fashion search, semantic search, vector embeddings, CLIP, image search, MongoDB Atlas',
  openGraph: {
    title: 'AI Fashion Search - Semantic Search Demo',
    description: 'Advanced fashion discovery using AI semantic search technology',
    type: 'website',
  }
} 