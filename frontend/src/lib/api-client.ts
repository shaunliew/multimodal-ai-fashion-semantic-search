/**
 * @fileoverview API client for FastAPI backend integration
 * @learning Demonstrates axios configuration, error handling, and TypeScript integration
 * @concepts HTTP clients, API abstraction, error handling, type safety
 */

import axios, { AxiosInstance, AxiosError } from 'axios'
import {
  TextSearchRequest,
  ImageSearchRequest,
  SearchResponse,
  HealthCheckResponse,
  StatsResponse,
  ErrorResponse,
  isErrorResponse
} from '@/types/api'

/**
 * Learning: Environment-based API URL configuration
 * In production, this would come from environment variables
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

/**
 * Custom error class for API errors
 * @learning Custom error classes provide better error handling and debugging
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public detail?: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * Axios instance with default configuration
 * @learning Centralized axios configuration ensures consistency
 */
const axiosInstance: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds for AI operations
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Request interceptor for logging and auth
 * @learning Interceptors provide centralized request/response handling
 */
axiosInstance.interceptors.request.use(
  (config) => {
    // Log requests in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`🚀 ${config.method?.toUpperCase()} ${config.url}`, config.data)
    }
    return config
  },
  (error) => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

/**
 * Response interceptor for error handling
 * @learning Centralized error handling improves consistency
 */
axiosInstance.interceptors.response.use(
  (response) => {
    // Log responses in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`✅ Response from ${response.config.url}:`, response.data)
    }
    return response
  },
  (error: AxiosError<ErrorResponse>) => {
    // Handle API errors
    if (error.response?.data && isErrorResponse(error.response.data)) {
      const apiError = new ApiError(
        error.response.data.error.detail,
        error.response.data.error.status_code,
        error.response.data.error.detail
      )
      return Promise.reject(apiError)
    }
    
    // Handle network errors
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new ApiError('Request timeout', 408))
    }
    
    if (!error.response) {
      return Promise.reject(new ApiError('Network error - API unavailable', 503))
    }
    
    // Generic error
    return Promise.reject(new ApiError(
      error.message,
      error.response?.status,
      error.response?.statusText
    ))
  }
)

/**
 * API client with methods for all endpoints
 * @learning Abstraction layer between UI and API implementation
 */
export const apiClient = {
  /**
   * Health check endpoint
   * @learning Health checks verify system readiness before operations
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await axiosInstance.get<HealthCheckResponse>('/health')
    return response.data
  },

  /**
   * Get system statistics
   * @learning Statistics help monitor system performance
   */
  async getStats(): Promise<StatsResponse> {
    const response = await axiosInstance.get<StatsResponse>('/stats')
    return response.data
  },

  /**
   * Text-to-image semantic search
   * @learning CLIP embeddings enable cross-modal search
   */
  async searchByText(request: TextSearchRequest): Promise<SearchResponse> {
    const response = await axiosInstance.post<SearchResponse>('/search/text', request)
    return response.data
  },

  /**
   * Image-to-image search with base64 or URL
   * @learning JSON endpoint for programmatic image search
   */
  async searchByImage(request: ImageSearchRequest): Promise<SearchResponse> {
    const response = await axiosInstance.post<SearchResponse>('/search/image', request)
    return response.data
  },

  /**
   * Image-to-image search with file upload
   * @learning Multipart form data for direct file uploads
   */
  async searchByImageFile(
    file: File,
    numberOfResults: number = 10,
    excludeReference: boolean = true
  ): Promise<SearchResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('number_of_results', numberOfResults.toString())
    formData.append('exclude_reference', excludeReference.toString())

    const response = await axiosInstance.post<SearchResponse>(
      '/search/image/v2',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return response.data
  },

  /**
   * Convert File to base64 for image search
   * @learning Base64 encoding enables image transmission in JSON
   */
  async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const base64 = reader.result as string
        resolve(base64)
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  },

  /**
   * Helper to check if API is available
   * @learning Quick connectivity check before operations
   */
  async isApiAvailable(): Promise<boolean> {
    try {
      const health = await this.healthCheck()
      return health.status === 'healthy'
    } catch {
      return false
    }
  },
}

/**
 * Export specific endpoint functions for direct use
 * @learning Named exports provide better tree-shaking
 */
export const {
  healthCheck,
  getStats,
  searchByText,
  searchByImage,
  searchByImageFile,
  fileToBase64,
  isApiAvailable,
} = apiClient 