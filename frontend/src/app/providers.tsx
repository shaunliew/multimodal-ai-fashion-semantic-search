/**
 * @fileoverview Client-side providers for the application
 * @learning React Query requires a QueryClient provider at the app root
 * @concepts Provider pattern, React Query setup, DevTools integration
 */

'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { useState } from 'react'

/**
 * App-wide providers component
 * @learning Providers enable global state and functionality across the app
 */
export function Providers({ children }: { children: React.ReactNode }) {
  /**
   * Create QueryClient instance
   * @learning QueryClient manages caching, background refetching, and more
   */
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Stale time: how long until data is considered stale
            staleTime: 60 * 1000, // 1 minute
            
            // GC time: how long to keep unused data in cache
            gcTime: 5 * 60 * 1000, // 5 minutes
            
            // Retry configuration for failed requests
            retry: (failureCount, error) => {
              // Don't retry on 4xx errors (client errors)
              if (error instanceof Error && 'status' in error) {
                const errorWithStatus = error as Error & { status?: number }
                if (errorWithStatus.status && errorWithStatus.status >= 400 && errorWithStatus.status < 500) {
                  return false
                }
              }
              // Retry up to 3 times for other errors
              return failureCount < 3
            },
            
            // Exponential backoff for retries
            retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
            
            // Refetch on window focus for fresh data
            refetchOnWindowFocus: false, // Disabled for AI search to avoid unwanted refetches
          },
          mutations: {
            // Retry configuration for mutations
            retry: 1,
            retryDelay: 1000,
          },
        },
      })
  )

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {/* React Query DevTools for debugging in development */}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools 
          initialIsOpen={false} 
          position="bottom"
        />
      )}
    </QueryClientProvider>
  )
} 