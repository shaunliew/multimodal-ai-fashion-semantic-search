/**
 * @fileoverview Download Store - Zustand state management for image download functionality
 * @learning Demonstrates centralized download state management with progress tracking and error handling
 * @concepts Download queues, progress tracking, error states, file operations, concurrent downloads, toast notifications
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { toast } from 'sonner'
import { downloadImage, generateImageFilename, getImageExtension } from '@/lib/downloadUtils'

/**
 * Learning: Download item interface for tracking individual download operations
 * Each download has its own state, progress, and metadata for better UX
 */
interface DownloadItem {
  id: string
  imageUrl: string
  title: string
  progress: number
  status: 'idle' | 'downloading' | 'completed' | 'error'
  error?: string
  startedAt?: Date
  completedAt?: Date
}

/**
 * Learning: Comprehensive download state interface
 * Centralizes all download-related state and operations for consistent behavior
 */
interface DownloadState {
  // Download items map (itemId -> DownloadItem)
  downloads: Record<string, DownloadItem>
  
  // Global download state
  isAnyDownloading: boolean
  totalActiveDownloads: number
  
  // Actions for download management
  startDownload: (itemId: string, imageUrl: string, title: string) => Promise<void>
  updateProgress: (itemId: string, progress: number) => void
  completeDownload: (itemId: string) => void
  failDownload: (itemId: string, error: string) => void
  cancelDownload: (itemId: string) => void
  clearDownload: (itemId: string) => void
  clearAllDownloads: () => void
  
  // Query functions
  getDownloadStatus: (itemId: string) => 'idle' | 'downloading' | 'completed' | 'error'
  getDownloadProgress: (itemId: string) => number
  isDownloading: (itemId: string) => boolean
  getDownloadError: (itemId: string) => string | undefined
  
  // Batch operations
  getActiveDownloads: () => DownloadItem[]
  getCompletedDownloads: () => DownloadItem[]
  getFailedDownloads: () => DownloadItem[]
}

/**
 * Learning: Helper function to get user-friendly error messages
 * Translates technical errors into actionable user feedback
 */
const getUserFriendlyErrorMessage = (error: string, title: string): string => {
  if (error.includes('Failed to fetch') || error.includes('CORS')) {
    return `Cannot download "${title}" - image is protected by CORS policy. This is common with external fashion sites.`
  }
  if (error.includes('Network request failed') || error.includes('network')) {
    return `Download failed for "${title}" - please check your internet connection and try again.`
  }
  if (error.includes('403') || error.includes('Forbidden')) {
    return `Access denied for "${title}" - the image server doesn't allow downloads.`
  }
  if (error.includes('404') || error.includes('Not Found')) {
    return `Image not found for "${title}" - the file may have been moved or deleted.`
  }
  return `Download failed for "${title}": ${error}`
}

/**
 * Learning: Zustand store for centralized download state management
 * Benefits over useState:
 * - Consistent download state across all components
 * - Centralized progress tracking and error handling
 * - Better debugging with DevTools integration
 * - No prop drilling for download state
 * - Concurrent download management
 * - Toast notifications for user feedback
 */
export const useDownloadStore = create<DownloadState>()(
  devtools(
    (set, get) => ({
      // Initial state
      downloads: {},
      isAnyDownloading: false,
      totalActiveDownloads: 0,

      // Start download action
      startDownload: async (itemId: string, imageUrl: string, title: string) => {
        const { downloads } = get()
        
        // Prevent duplicate downloads
        if (downloads[itemId]?.status === 'downloading') {
          toast.info(`Download already in progress for "${title}"`)
          return
        }

        /**
         * Learning: Create download item with initial state
         * Track metadata for better UX and debugging
         */
        const downloadItem: DownloadItem = {
          id: itemId,
          imageUrl,
          title,
          progress: 0,
          status: 'downloading',
          startedAt: new Date()
        }

        // Update state with new download
        set((state) => ({
          downloads: {
            ...state.downloads,
            [itemId]: downloadItem
          },
          isAnyDownloading: true,
          totalActiveDownloads: state.totalActiveDownloads + 1
        }))

        // Show starting toast
        toast.loading(`Downloading "${title}"...`, {
          id: `download-${itemId}`,
          description: 'Preparing your image download'
        })

        try {
          // Generate proper filename
          const extension = getImageExtension(imageUrl)
          const filename = generateImageFilename(itemId, title, extension)

          // Start actual download with progress callbacks
          await downloadImage(
            imageUrl,
            filename,
            // Progress callback - updates store state
            (progress) => {
              get().updateProgress(itemId, progress)
              
              // Update toast with progress for slow downloads
              if (progress > 10 && progress < 100) {
                toast.loading(`Downloading "${title}" (${progress}%)`, {
                  id: `download-${itemId}`,
                  description: 'Your download is in progress'
                })
              }
            },
            // Error callback - updates store with error
            (error) => {
              get().failDownload(itemId, error)
            }
          )

          // Mark as completed if no errors thrown
          get().completeDownload(itemId)

        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown download error'
          get().failDownload(itemId, errorMessage)
        }
      },

      // Update progress for specific download
      updateProgress: (itemId: string, progress: number) => {
        set((state) => ({
          downloads: {
            ...state.downloads,
            [itemId]: {
              ...state.downloads[itemId],
              progress: Math.min(Math.max(progress, 0), 100) // Clamp between 0-100
            }
          }
        }))
      },

      // Complete download successfully
      completeDownload: (itemId: string) => {
        set((state) => {
          const download = state.downloads[itemId]
          if (!download) return state

          const updatedDownload = {
            ...download,
            status: 'completed' as const,
            progress: 100,
            completedAt: new Date()
          }

          const newActiveCount = state.totalActiveDownloads - 1

          // Show success toast
          toast.success(`Downloaded "${download.title}" successfully!`, {
            id: `download-${itemId}`,
            description: 'Image saved to your downloads folder',
            action: {
              label: 'Open Downloads',
              onClick: () => {
                // This will open the downloads folder in most browsers
                const link = document.createElement('a')
                link.href = 'file://' + (navigator.platform.includes('Win') ? 'C:/Users/' + (process.env.USERNAME || 'User') + '/Downloads' : '/Users/' + (process.env.USER || 'user') + '/Downloads')
                link.target = '_blank'
                document.body.appendChild(link)
                link.click()
                document.body.removeChild(link)
              }
            }
          })

          return {
            downloads: {
              ...state.downloads,
              [itemId]: updatedDownload
            },
            totalActiveDownloads: newActiveCount,
            isAnyDownloading: newActiveCount > 0
          }
        })

        // Auto-clear completed downloads after delay
        setTimeout(() => {
          get().clearDownload(itemId)
        }, 5000) // Extended to 5 seconds to let user see success message
      },

      // Mark download as failed
      failDownload: (itemId: string, error: string) => {
        set((state) => {
          const download = state.downloads[itemId]
          if (!download) return state

          const updatedDownload = {
            ...download,
            status: 'error' as const,
            error,
            completedAt: new Date()
          }

          const newActiveCount = state.totalActiveDownloads - 1

          // Show error toast with user-friendly message
          const userFriendlyMessage = getUserFriendlyErrorMessage(error, download.title)
          
          toast.error('Download Failed', {
            id: `download-${itemId}`,
            description: userFriendlyMessage,
            action: {
              label: 'Retry',
              onClick: () => {
                // Retry the download
                get().startDownload(itemId, download.imageUrl, download.title)
              }
            },
            duration: 10000 // Show error longer for user to read
          })

          return {
            downloads: {
              ...state.downloads,
              [itemId]: updatedDownload
            },
            totalActiveDownloads: newActiveCount,
            isAnyDownloading: newActiveCount > 0
          }
        })

        // Auto-clear failed downloads after longer delay
        setTimeout(() => {
          get().clearDownload(itemId)
        }, 15000) // Extended to 15 seconds for error messages
      },

      // Cancel ongoing download
      cancelDownload: (itemId: string) => {
        set((state) => {
          const download = state.downloads[itemId]
          if (!download || download.status !== 'downloading') return state

          const newActiveCount = state.totalActiveDownloads - 1

          // Dismiss the download toast
          toast.dismiss(`download-${itemId}`)
          
          // Show cancellation message
          toast.info(`Download cancelled for "${download.title}"`)

          return {
            downloads: {
              ...state.downloads,
              [itemId]: {
                ...download,
                status: 'idle',
                progress: 0
              }
            },
            totalActiveDownloads: newActiveCount,
            isAnyDownloading: newActiveCount > 0
          }
        })
      },

      // Clear specific download from state
      clearDownload: (itemId: string) => {
        set((state) => {
          const remainingDownloads = { ...state.downloads }
          delete remainingDownloads[itemId]
          
          // Dismiss any remaining toast for this download
          toast.dismiss(`download-${itemId}`)
          
          return {
            downloads: remainingDownloads
          }
        })
      },

      // Clear all downloads
      clearAllDownloads: () => {
        // Dismiss all download toasts
        Object.keys(get().downloads).forEach(itemId => {
          toast.dismiss(`download-${itemId}`)
        })
        
        set({
          downloads: {},
          isAnyDownloading: false,
          totalActiveDownloads: 0
        })
        
        toast.info('All downloads cleared')
      },

      // Query functions for components
      getDownloadStatus: (itemId: string) => {
        const download = get().downloads[itemId]
        return download?.status || 'idle'
      },

      getDownloadProgress: (itemId: string) => {
        const download = get().downloads[itemId]
        return download?.progress || 0
      },

      isDownloading: (itemId: string) => {
        const download = get().downloads[itemId]
        return download?.status === 'downloading'
      },

      getDownloadError: (itemId: string) => {
        const download = get().downloads[itemId]
        return download?.error
      },

      // Batch query functions
      getActiveDownloads: () => {
        const downloads = get().downloads
        return Object.values(downloads).filter(d => d.status === 'downloading')
      },

      getCompletedDownloads: () => {
        const downloads = get().downloads
        return Object.values(downloads).filter(d => d.status === 'completed')
      },

      getFailedDownloads: () => {
        const downloads = get().downloads
        return Object.values(downloads).filter(d => d.status === 'error')
      }
    }),
    { 
      name: 'download-store', // DevTools identifier
      /**
       * Learning: Store separation for download functionality
       * Download state is separate from other app state for better organization
       * Now includes comprehensive toast notification integration
       */
    }
  )
) 