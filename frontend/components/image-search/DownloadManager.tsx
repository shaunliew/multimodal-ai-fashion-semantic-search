/**
 * @fileoverview Download Manager - Global download status and control component
 * @learning Demonstrates centralized download state visualization and batch operations
 * @concepts Zustand global state access, download queues, batch operations, user feedback
 */

'use client'

import React from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { 
  Download, 
  X, 
  CheckCircle, 
  AlertCircle,
  Loader2,
  Trash2
} from 'lucide-react'
import { useDownloadStore } from '@/stores/downloadStore'
import { cn } from '@/lib/utils'

interface DownloadManagerProps {
  className?: string
}

/**
 * Learning: Global download manager using Zustand store
 * This component demonstrates how centralized state allows for:
 * - Real-time download monitoring across the app
 * - Batch operations on downloads
 * - Consistent UI updates without prop drilling
 */
export const DownloadManager = ({ className }: DownloadManagerProps) => {
  /**
   * Learning: Using Zustand store selectors for efficient re-renders
   * Only re-renders when relevant download state changes
   */
  const { 
    downloads,
    isAnyDownloading,
    totalActiveDownloads,
    getActiveDownloads,
    getCompletedDownloads,
    getFailedDownloads,
    cancelDownload,
    clearDownload,
    clearAllDownloads,
    startDownload
  } = useDownloadStore()

  const activeDownloads = getActiveDownloads()
  const completedDownloads = getCompletedDownloads()
  const failedDownloads = getFailedDownloads()
  const totalDownloads = Object.keys(downloads).length

  /**
   * Learning: Early return pattern for cleaner component logic
   * Don't render anything if there are no downloads to show
   */
  if (totalDownloads === 0) {
    return null
  }

  /**
   * Learning: Action handlers using Zustand store actions
   * All state mutations go through the centralized store
   */
  const handleCancelDownload = (itemId: string) => {
    cancelDownload(itemId)
  }

  const handleRetryDownload = (itemId: string) => {
    const download = downloads[itemId]
    if (download) {
      startDownload(itemId, download.imageUrl, download.title)
    }
  }

  const handleClearDownload = (itemId: string) => {
    clearDownload(itemId)
  }

  const handleClearAll = () => {
    clearAllDownloads()
  }

  return (
    <Card className={cn("w-full max-w-md", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Download className="h-4 w-4" />
            Downloads
            {totalDownloads > 0 && (
              <Badge variant="secondary" className="text-xs">
                {totalDownloads}
              </Badge>
            )}
          </CardTitle>
          
          {totalDownloads > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearAll}
              className="h-7 text-xs"
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Clear All
            </Button>
          )}
        </div>
        
        {/* Global Progress Summary */}
        {isAnyDownloading && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{totalActiveDownloads} active download{totalActiveDownloads !== 1 ? 's' : ''}</span>
              <span>In progress...</span>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-3 max-h-96 overflow-y-auto">
        {/* Active Downloads */}
        {activeDownloads.map((download) => (
          <DownloadItem
            key={download.id}
            download={download}
            onCancel={() => handleCancelDownload(download.id)}
          />
        ))}

        {/* Completed Downloads */}
        {completedDownloads.map((download) => (
          <DownloadItem
            key={download.id}
            download={download}
            onClear={() => handleClearDownload(download.id)}
          />
        ))}

        {/* Failed Downloads */}
        {failedDownloads.map((download) => (
          <DownloadItem
            key={download.id}
            download={download}
            onRetry={() => handleRetryDownload(download.id)}
            onClear={() => handleClearDownload(download.id)}
          />
        ))}
      </CardContent>
    </Card>
  )
}

/**
 * Individual download item component
 */
interface DownloadItemProps {
  download: {
    id: string
    title: string
    progress: number
    status: 'idle' | 'downloading' | 'completed' | 'error'
    error?: string
  }
  onCancel?: () => void
  onRetry?: () => void
  onClear?: () => void
}

const DownloadItem = ({ download, onCancel, onRetry, onClear }: DownloadItemProps) => {
  const getStatusIcon = () => {
    switch (download.status) {
      case 'downloading':
        return <Loader2 className="h-3 w-3 animate-spin text-blue-500" />
      case 'completed':
        return <CheckCircle className="h-3 w-3 text-green-500" />
      case 'error':
        return <AlertCircle className="h-3 w-3 text-red-500" />
      default:
        return <Download className="h-3 w-3 text-muted-foreground" />
    }
  }

  const getStatusColor = () => {
    switch (download.status) {
      case 'downloading':
        return 'border-blue-200 bg-blue-50'
      case 'completed':
        return 'border-green-200 bg-green-50'
      case 'error':
        return 'border-red-200 bg-red-50'
      default:
        return 'border-gray-200 bg-gray-50'
    }
  }

  return (
    <div className={cn(
      "flex items-center gap-3 p-3 rounded-lg border transition-colors",
      getStatusColor()
    )}>
      {/* Status Icon */}
      <div className="flex-shrink-0">
        {getStatusIcon()}
      </div>

      {/* Download Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <p className="text-xs font-medium truncate" title={download.title}>
            {download.title}
          </p>
          <span className="text-xs text-muted-foreground ml-2">
            {download.status === 'downloading' && `${download.progress}%`}
            {download.status === 'completed' && 'Done'}
            {download.status === 'error' && 'Failed'}
          </span>
        </div>

        {/* Progress Bar for Active Downloads */}
        {download.status === 'downloading' && (
          <Progress value={download.progress} className="h-1 mb-2" />
        )}

        {/* Error Message */}
        {download.status === 'error' && download.error && (
          <p className="text-xs text-red-600 mb-2 line-clamp-2">
            {download.error}
          </p>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-1">
        {download.status === 'downloading' && onCancel && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onCancel}
            className="h-6 w-6 p-0"
            title="Cancel download"
          >
            <X className="h-3 w-3" />
          </Button>
        )}

        {download.status === 'error' && onRetry && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRetry}
            className="h-6 w-6 p-0"
            title="Retry download"
          >
            <Download className="h-3 w-3" />
          </Button>
        )}

        {(download.status === 'completed' || download.status === 'error') && onClear && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            className="h-6 w-6 p-0"
            title="Clear from list"
          >
            <X className="h-3 w-3" />
          </Button>
        )}
      </div>
    </div>
  )
}

export { DownloadItem } 