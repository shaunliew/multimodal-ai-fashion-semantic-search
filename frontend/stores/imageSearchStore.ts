/**
 * @fileoverview Image Search Store - Zustand state management for semantic search
 * @learning Demonstrates centralized state management for AI-powered search functionality
 * @concepts Vector embeddings, similarity scoring, search modes, optimistic updates
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

/**
 * Learning: Search modes explain different approaches to semantic search
 * - TEXT_TO_IMAGE: Natural language → visual embeddings (CLIP-style)
 * - IMAGE_TO_IMAGE: Visual similarity using computer vision models
 * - HYBRID: Combines text and visual features for enhanced accuracy
 */
export const SEARCH_MODES = {
  TEXT_TO_IMAGE: 'text_to_image',
  IMAGE_TO_IMAGE: 'image_to_image', 
  HYBRID: 'hybrid_search'
} as const

export type SearchMode = typeof SEARCH_MODES[keyof typeof SEARCH_MODES]

/**
 * Learning: Search result structure for vector-based image search
 * - similarityScore: Cosine similarity (0-1) from vector embedding comparison
 * - embedding: High-dimensional vector representation of image content
 * - metadata: Additional context for search ranking and filtering
 */
export interface SearchResult {
  id: string
  imageUrl: string
  title: string
  description: string
  similarityScore: number // 0-1 range, higher = more similar
  embedding?: number[] // Optional: for advanced similarity visualization
  metadata: {
    tags: string[]
    category: string
    source: string
    uploadedAt: string
  }
}

/**
 * Learning: Search filters for refining vector search results
 * Vector search is computationally expensive, so filtering helps focus results
 */
export interface SearchFilters {
  category?: string
  minSimilarity: number // Threshold for relevance (e.g., 0.7 = 70% similar)
  maxResults: number // Limit results for performance
  tags?: string[]
}

interface ImageSearchState {
  // Search query state
  textQuery: string
  uploadedImage: File | null
  searchMode: SearchMode
  
  // Results and UI state
  results: SearchResult[]
  isSearching: boolean
  error: string | null
  
  // Search configuration
  filters: SearchFilters
  
  // Actions with clear educational purpose
  setTextQuery: (query: string) => void
  setUploadedImage: (file: File | null) => void
  setSearchMode: (mode: SearchMode) => void
  setFilters: (filters: Partial<SearchFilters>) => void
  
  // Core search functionality
  performTextSearch: (query: string) => Promise<void>
  performImageSearch: (imageFile: File) => Promise<void>
  clearResults: () => void
  clearError: () => void
}

/**
 * Learning: Zustand provides simple, TypeScript-first state management
 * Benefits over Redux:
 * - Less boilerplate code
 * - Better TypeScript integration
 * - No providers needed
 * - Simpler async actions
 */
export const useImageSearchStore = create<ImageSearchState>()(
  devtools(
    (set, get) => ({
      // Initial state
      textQuery: '',
      uploadedImage: null,
      searchMode: SEARCH_MODES.TEXT_TO_IMAGE,
      results: [],
      isSearching: false,
      error: null,
      filters: {
        minSimilarity: 0.7, // 70% similarity threshold
        maxResults: 20,
        category: undefined,
        tags: undefined
      },

      // Action: Update text query
      setTextQuery: (query) => {
        set({ textQuery: query, error: null })
      },

      // Action: Handle image upload for image-to-image search
      setUploadedImage: (file) => {
        set({ 
          uploadedImage: file, 
          error: null,
          // Auto-switch to image search mode when image is uploaded
          searchMode: file ? SEARCH_MODES.IMAGE_TO_IMAGE : get().searchMode
        })
      },

      // Action: Switch between search modes
      setSearchMode: (mode) => {
        set({ searchMode: mode, error: null })
        
        /**
         * Learning: Clear irrelevant state when switching modes
         * This prevents confusion and improves UX
         */
        if (mode === SEARCH_MODES.TEXT_TO_IMAGE) {
          set({ uploadedImage: null })
        }
      },

      // Action: Update search filters
      setFilters: (newFilters) => {
        set({ 
          filters: { ...get().filters, ...newFilters },
          error: null 
        })
        
        /**
         * Learning: Auto-trigger search when filters change
         * This provides immediate feedback for filter adjustments
         */
        const { textQuery, uploadedImage, searchMode } = get()
        if (searchMode === SEARCH_MODES.TEXT_TO_IMAGE && textQuery) {
          get().performTextSearch(textQuery)
        } else if (searchMode === SEARCH_MODES.IMAGE_TO_IMAGE && uploadedImage) {
          get().performImageSearch(uploadedImage)
        }
      },

      // Core Action: Perform text-to-image search
      performTextSearch: async (query) => {
        if (!query.trim()) {
          set({ error: 'Please enter a search query' })
          return
        }

        set({ isSearching: true, error: null })
        
        try {
          /**
           * Learning: This simulates API call to backend vector search
           * In production, this would:
           * 1. Convert text to embeddings using CLIP or similar model
           * 2. Query vector database (MongoDB Atlas Vector Search)
           * 3. Return results ranked by cosine similarity
           */
          await simulateVectorSearch('text')
          
          // Mock results for demonstration
          const mockResults: SearchResult[] = Array.from({ length: 8 }, (_, i) => ({
            id: `result-${i}`,
            imageUrl: `https://picsum.photos/300/400?random=${i}&query=${encodeURIComponent(query)}`,
            title: `Fashion Item ${i + 1}`,
            description: `${query} - Similar fashion item with AI-detected visual features`,
            similarityScore: Math.random() * 0.4 + 0.6, // 0.6-1.0 range
            metadata: {
              tags: ['fashion', 'clothing', query.toLowerCase()],
              category: 'apparel',
              source: 'demo_dataset',
              uploadedAt: new Date().toISOString()
            }
          })).filter(result => result.similarityScore >= get().filters.minSimilarity)
          
          set({ 
            results: mockResults.slice(0, get().filters.maxResults),
            isSearching: false 
          })
          
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Search failed',
            isSearching: false,
            results: []
          })
        }
      },

      // Core Action: Perform image-to-image search
      performImageSearch: async (imageFile) => {
        if (!imageFile) {
          set({ error: 'Please upload an image' })
          return
        }

        set({ isSearching: true, error: null })
        
        try {
          /**
           * Learning: Image-to-image search process:
           * 1. Extract visual features from uploaded image using CNN
           * 2. Convert to embedding vector (e.g., ResNet, EfficientNet features)
           * 3. Find similar vectors in database using cosine similarity
           * 4. Return visually similar images ranked by similarity score
           */
          await simulateVectorSearch('image')
          
          // Mock results for demonstration
          const mockResults: SearchResult[] = Array.from({ length: 12 }, (_, i) => ({
            id: `image-result-${i}`,
            imageUrl: `https://picsum.photos/300/400?random=${i + 20}`,
            title: `Similar Image ${i + 1}`,
            description: `Visually similar to uploaded image - detected features match`,
            similarityScore: Math.random() * 0.5 + 0.5, // 0.5-1.0 range
            metadata: {
              tags: ['similar', 'visual-match'],
              category: 'fashion',
              source: 'image_dataset',
              uploadedAt: new Date().toISOString()
            }
          })).filter(result => result.similarityScore >= get().filters.minSimilarity)
          
          set({ 
            results: mockResults
              .sort((a, b) => b.similarityScore - a.similarityScore) // Sort by similarity
              .slice(0, get().filters.maxResults),
            isSearching: false 
          })
          
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Image search failed',
            isSearching: false,
            results: []
          })
        }
      },

      // Utility Actions
      clearResults: () => {
        set({ results: [], error: null })
      },

      clearError: () => {
        set({ error: null })
      }
    }),
    { name: 'image-search-store' } // DevTools identifier
  )
)

/**
 * Learning: Simulate network delay for realistic UX testing
 * In production, this represents the time for:
 * - Embedding generation (100-500ms)
 * - Vector database query (50-200ms)
 * - Result processing and ranking (50-100ms)
 */
const simulateVectorSearch = (type: 'text' | 'image'): Promise<void> => {
  return new Promise((resolve) => {
    // Simulate realistic search latency
    const delay = type === 'image' ? 1500 : 1000 // Image processing takes longer
    setTimeout(resolve, delay)
  })
} 