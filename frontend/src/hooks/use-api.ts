/**
 * @fileoverview React Query hooks for API integration
 * @learning Demonstrates React Query patterns for data fetching, caching, and state management
 * @concepts Custom hooks, React Query, error handling, optimistic updates
 */

import {
  useQuery,
  useMutation,
  UseQueryResult,
  UseMutationResult,
} from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  TextSearchRequest,
  ImageSearchRequest,
  SearchResponse,
  HealthCheckResponse,
  StatsResponse,
} from '@/types/api'
import {
  apiClient,
  ApiError,
  searchByText,
  searchByImage,
  searchByImageFile,
  healthCheck,
  getStats,
} from '@/lib/api-client'
import { useState } from 'react'

/**
 * Query keys for React Query cache management
 * @learning Consistent query keys enable cache invalidation and refetching
 */
export const queryKeys = {
  health: ['health'] as const,
  stats: ['stats'] as const,
  textSearch: (query: string) => ['search', 'text', query] as const,
  imageSearch: (imageId: string) => ['search', 'image', imageId] as const,
}

/**
 * Hook for health check with auto-refresh
 * @learning Health checks ensure system readiness before user operations
 */
export function useHealthCheck(options?: {
  refetchInterval?: number
  enabled?: boolean
}): UseQueryResult<HealthCheckResponse, ApiError> {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: healthCheck,
    refetchInterval: options?.refetchInterval ?? 30000, // Check every 30 seconds
    enabled: options?.enabled ?? true,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })
}

/**
 * Hook for system statistics
 * @learning Statistics provide insights into system performance and data coverage
 */
export function useStats(options?: {
  enabled?: boolean
}): UseQueryResult<StatsResponse, ApiError> {
  return useQuery({
    queryKey: queryKeys.stats,
    queryFn: getStats,
    staleTime: 60000, // Consider fresh for 1 minute
    gcTime: 300000, // Keep in cache for 5 minutes (was cacheTime in v4)
    enabled: options?.enabled ?? true,
  })
}

/**
 * Hook for text-to-image search
 * @learning Mutations are used for search operations that modify server state
 */
export function useTextSearch(): UseMutationResult<
  SearchResponse,
  ApiError,
  TextSearchRequest
> {
  return useMutation({
    mutationFn: searchByText,
    onSuccess: (data) => {
      if (data.results_count === 0) {
        toast.info('No results found. Try different keywords.')
      } else {
        toast.success(`Found ${data.results_count} similar products!`)
      }
    },
    onError: (error: ApiError) => {
      console.error('Text search error:', error)
      toast.error(error.detail || 'Search failed. Please try again.')
    },
  })
}

/**
 * Hook for image-to-image search with base64/URL
 * @learning Supports multiple image input methods for flexibility
 */
export function useImageSearch(): UseMutationResult<
  SearchResponse,
  ApiError,
  ImageSearchRequest
> {
  return useMutation({
    mutationFn: searchByImage,
    onSuccess: (data) => {
      if (data.results_count === 0) {
        toast.info('No similar products found.')
      } else {
        toast.success(`Found ${data.results_count} visually similar products!`)
      }
    },
    onError: (error: ApiError) => {
      console.error('Image search error:', error)
      toast.error(error.detail || 'Image search failed. Please try again.')
    },
  })
}

/**
 * Hook for image-to-image search with file upload
 * @learning File uploads require different handling than JSON requests
 */
export function useImageFileSearch(): UseMutationResult<
  SearchResponse,
  ApiError,
  {
    file: File
    numberOfResults?: number
    excludeReference?: boolean
  }
> {
  return useMutation({
    mutationFn: ({ file, numberOfResults, excludeReference }) =>
      searchByImageFile(file, numberOfResults, excludeReference),
    onSuccess: (data) => {
      if (data.results_count === 0) {
        toast.info('No similar products found.')
      } else {
        toast.success(`Found ${data.results_count} visually similar products!`)
      }
      
      // Log performance metrics in development
      if (process.env.NODE_ENV === 'development') {
        console.log('Search performance:', data.performance)
        if (data.similarity_stats) {
          console.log('Similarity stats:', data.similarity_stats)
        }
      }
    },
    onError: (error: ApiError) => {
      console.error('Image file search error:', error)
      
      // Provide specific error messages
      if (error.status === 413) {
        toast.error('Image file too large. Please use a smaller image.')
      } else if (error.status === 415) {
        toast.error('Unsupported image format. Please use JPEG or PNG.')
      } else {
        toast.error(error.detail || 'Image search failed. Please try again.')
      }
    },
  })
}

/**
 * Combined search hook that handles both text and image search
 * @learning Unified interface simplifies component logic
 */
export function useSemanticSearch() {
  const textSearch = useTextSearch()
  const imageSearch = useImageSearch()
  const imageFileSearch = useImageFileSearch()
  const health = useHealthCheck({ refetchInterval: 60000 })
  const stats = useStats()

  /**
   * Learning: Track when a new search is starting to clear previous results
   * This prevents showing stale data during new search operations
   */
  const [clearingResults, setClearingResults] = useState(false)

  /**
   * Perform search based on input type
   * @learning Polymorphic search function adapts to input type
   */
  const search = async (params: {
    mode: 'text' | 'image' | 'url'
    query?: string
    imageFile?: File
    imageUrl?: string
    numberOfResults?: number
    excludeReference?: boolean
  }) => {
    // Check API health before searching
    if (health.data?.status !== 'healthy') {
      toast.error('Search service is currently unavailable. Please try again later.')
      return null
    }

    /**
     * Learning: Clear previous results immediately when new search starts
     * We set clearingResults to true to hide old data during the new search
     */
    setClearingResults(true)

    // Reset all mutation states to clear previous results
    textSearch.reset()
    imageSearch.reset()
    imageFileSearch.reset()

    try {
      let searchPromise: Promise<SearchResponse>

      switch (params.mode) {
        case 'text':
          if (!params.query) {
            toast.error('Please enter a search query')
            setClearingResults(false)
            return null
          }
          searchPromise = textSearch.mutateAsync({
            search_query: params.query,
            number_of_results: params.numberOfResults,
          })
          break

        case 'image':
          if (!params.imageFile) {
            toast.error('Please select an image')
            setClearingResults(false)
            return null
          }
          searchPromise = imageFileSearch.mutateAsync({
            file: params.imageFile,
            numberOfResults: params.numberOfResults,
            excludeReference: params.excludeReference,
          })
          break

        case 'url':
          if (!params.imageUrl) {
            toast.error('Please enter an image URL')
            setClearingResults(false)
            return null
          }
          searchPromise = imageSearch.mutateAsync({
            image_url: params.imageUrl,
            number_of_results: params.numberOfResults,
            exclude_reference: params.excludeReference,
          })
          break

        default:
          toast.error('Invalid search mode')
          setClearingResults(false)
          return null
      }

      const result = await searchPromise
      setClearingResults(false)
      return result

    } catch (error) {
      setClearingResults(false)
      throw error
    }
  }

  // Combine loading states - include clearingResults to show loading during transition
  const isSearching = clearingResults || textSearch.isPending || imageSearch.isPending || imageFileSearch.isPending

  // Get the most recent search results - return null if we're clearing results
  const results = clearingResults ? null : (textSearch.data || imageSearch.data || imageFileSearch.data)

  return {
    search,
    isSearching,
    results,
    health: health.data,
    stats: stats.data,
    isHealthy: health.data?.status === 'healthy',
    error: textSearch.error || imageSearch.error || imageFileSearch.error,
  }
}

/**
 * Hook for converting file to base64
 * @learning Utility hook for image preprocessing
 */
export function useFileToBase64() {
  return useMutation({
    mutationFn: apiClient.fileToBase64,
    onError: (error) => {
      console.error('File conversion error:', error)
      toast.error('Failed to process image file')
    },
  })
} 