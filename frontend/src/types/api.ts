/**
 * @fileoverview API type definitions matching FastAPI backend
 * @learning These types ensure type safety between frontend and backend
 * @concepts TypeScript interfaces, API contracts, type safety
 */

// ===========================
// Request Types
// ===========================

/**
 * Text search request matching FastAPI TextSearchRequest
 * @learning Pydantic models in backend map to TypeScript interfaces
 */
export interface TextSearchRequest {
  search_query: string
  number_of_results?: number
  similarity_threshold?: number
  filters?: Record<string, string | number | boolean>
}

/**
 * Image search request for JSON endpoint
 * @learning Supports both base64 and URL image inputs
 */
export interface ImageSearchRequest {
  input_image?: string  // Base64-encoded image
  image_url?: string    // URL of reference image
  number_of_results?: number
  exclude_reference?: boolean
}

/**
 * Image search request for multipart form endpoint
 * @learning Form data requires different handling than JSON
 */
export interface ImageSearchFormRequest {
  file: File
  number_of_results?: number
  exclude_reference?: boolean
}

// ===========================
// Response Types
// ===========================

/**
 * Fashion product model with comprehensive metadata
 * @learning Rich metadata enables filtering and contextual understanding
 */
export interface FashionProduct {
  id: string
  name: string
  brand?: string
  image: string
  
  // Fashion-specific attributes
  article_type?: string
  master_category?: string
  sub_category?: string
  gender?: string
  base_colour?: string
  season?: string
  year?: number
  usage?: string
  
  // Search metadata
  similarity_score: number
  description?: string
}

/**
 * Unified search response model
 * @learning Consistent response structure simplifies client implementation
 */
export interface SearchResponse {
  success: boolean
  query_type: 'text-to-image' | 'image-to-image'
  results_count: number
  total_results?: number
  results: FashionProduct[]
  
  // Performance metrics
  performance: {
    embedding_generation?: number
    vector_search?: number
    result_formatting?: number
    total_time: number
  }
  
  // Search quality metrics
  similarity_stats?: {
    max: number
    min: number
    average: number
    std_dev: number
  }
  
  // Pagination info
  pagination?: {
    requested: number
    returned: number
    limit_applied: boolean
    reason: string
  }
  
  // Image search specific
  image_source?: 'base64' | 'url' | 'file_upload' | 'unknown'
  
  message?: string
}

/**
 * Health check response
 * @learning Health checks verify system readiness
 */
export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy'
  timestamp: string
  components: {
    mongodb?: {
      status: string
      latency_ms?: number
      total_products?: number
      products_with_embeddings?: number
      coverage_percentage?: number
      error?: string
    }
    clip_model?: {
      status: string
      latency_ms?: number
      device?: string
      embedding_dimension?: number
      model?: string
      error?: string
    }
    vector_search?: {
      status: string
      latency_ms?: number
      index_name?: string
      test_results_found?: boolean
      error?: string
      reason?: string
    }
  }
  message: string
}

/**
 * Statistics response
 * @learning System statistics help monitor performance and data quality
 */
export interface StatsResponse {
  database_stats: {
    total_products: number
    products_with_embeddings: number
    embedding_coverage: number
    database_name: string
    collection_name: string
  }
  system_info: {
    vector_index: string
    embedding_model: string
    embedding_dimensions: number
    device: string
    similarity_metric: string
  }
  data_distribution: {
    top_categories: Array<{
      category: string
      count: number
    }>
    gender_distribution: Array<{
      gender: string
      count: number
    }>
  }
}

/**
 * Error response structure
 * @learning Consistent error handling improves debugging
 */
export interface ErrorResponse {
  success: false
  error: {
    type: string
    status_code: number
    detail: string
  }
  timestamp: string
}

// ===========================
// Utility Types
// ===========================

/**
 * API response wrapper for type guards
 * @learning Type guards ensure runtime type safety
 */
export type ApiResponse<T> = T | ErrorResponse

/**
 * Type guard to check if response is an error
 * @learning Runtime type checking for API responses
 */
export function isErrorResponse(response: unknown): response is ErrorResponse {
  return response !== null && 
    typeof response === 'object' &&
    'success' in response &&
    response.success === false && 
    'error' in response
}

/**
 * Search mode types for UI
 * @learning Discriminated unions provide type safety for different search modes
 */
export type SearchMode = 'text' | 'image' | 'url'

/**
 * Search state for UI management
 * @learning Centralized state type for search functionality
 */
export interface SearchState {
  mode: SearchMode
  query: string
  imageFile?: File
  imageUrl?: string
  imagePreview?: string
  isSearching: boolean
  results: FashionProduct[]
  error?: string
  performance?: SearchResponse['performance']
  stats?: SearchResponse['similarity_stats']
} 