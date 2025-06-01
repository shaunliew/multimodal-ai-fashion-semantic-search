/**
 * @fileoverview Image Zoom Modal - Fullscreen image viewer with Zustand state management
 * @learning Demonstrates responsive image viewing with centralized Zustand state management
 * @concepts Zustand integration, image optimization, touch interactions, responsive design, accessibility
 */

'use client'

import React, { useRef } from 'react'
import Image from 'next/image'
import { 
  Dialog, 
  DialogContent,
  DialogClose,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  X, 
  ZoomIn, 
  ZoomOut, 
  RotateCw, 
  Download,
  Heart,
  Move,
  Loader2
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLikedItemsStore } from '@/stores/likedItemsStore'
import { useImageZoomStore } from '@/stores/imageZoomStore'
import { useDownloadStore } from '@/stores/downloadStore'

interface ImageZoomModalProps {
  onDownload?: () => void
}

/**
 * Learning: Simplified zoom modal using Zustand for all state management
 * This demonstrates how centralized state management reduces component complexity
 * and eliminates prop drilling for modal state - now includes download state management
 * Enhanced with proper accessibility using DialogHeader and DialogTitle
 */
export const ImageZoomModal = ({ onDownload }: ImageZoomModalProps) => {
  // Refs for image container
  const imageContainerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  
  /**
   * Learning: Using multiple Zustand stores for different concerns
   * - useImageZoomStore: Zoom and pan functionality
   * - useLikedItemsStore: Favorites functionality
   * - useDownloadStore: Download state and progress tracking
   * This demonstrates clean separation of concerns with Zustand
   */
  const {
    // State
    isOpen,
    zoom,
    rotation,
    position,
    isDragging,
    currentImageUrl,
    currentImageAlt,
    currentItemId,
    currentItemTitle,
    
    // Actions
    zoomIn,
    zoomOut,
    rotate,
    resetView,
    closeZoomModal,
    startDrag,
    updateDragPosition,
    stopDrag,
    handleWheelZoom
  } = useImageZoomStore()
  
  // Liked items integration
  const { isLiked, toggleLike } = useLikedItemsStore()
  const isItemLiked = isLiked(currentItemId)

  /**
   * Learning: Download store integration replaces useState hooks
   * All download state is now managed centrally with better consistency
   */
  const { 
    startDownload, 
    isDownloading, 
    getDownloadProgress, 
    getDownloadStatus 
  } = useDownloadStore()

  // Get download state for current item
  const itemIsDownloading = isDownloading(currentItemId)
  const downloadProgress = getDownloadProgress(currentItemId)
  const downloadStatus = getDownloadStatus(currentItemId)

  /**
   * Learning: Event handlers using Zustand actions
   * All state mutations go through the store, maintaining consistency
   */
  const handleMouseDown = (e: React.MouseEvent) => {
    startDrag({ x: e.clientX, y: e.clientY })
    e.preventDefault()
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      updateDragPosition({ x: e.clientX, y: e.clientY })
    }
  }

  const handleMouseUp = () => {
    stopDrag()
  }

  /**
   * Learning: Touch events for mobile support
   * Zustand actions work seamlessly with touch interactions
   */
  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      const touch = e.touches[0]
      startDrag({ x: touch.clientX, y: touch.clientY })
    }
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 1 && isDragging) {
      const touch = e.touches[0]
      updateDragPosition({ x: touch.clientX, y: touch.clientY })
      e.preventDefault()
    }
  }

  const handleTouchEnd = () => {
    stopDrag()
  }

  /**
   * Learning: Wheel zoom using Zustand action
   * Centralized logic makes wheel zoom behavior consistent
   */
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    handleWheelZoom(e.deltaY)
  }

  /**
   * Learning: Like functionality using item data from store
   * Store provides all necessary item information for like action
   */
  const handleLike = () => {
    toggleLike(currentItemId, {
      imageUrl: currentImageUrl,
      title: currentItemTitle || 'Untitled Item'
    })
  }

  /**
   * Learning: Enhanced download using Zustand store action
   * All download logic is now centralized in the store with automatic state management
   */
  const handleDownloadClick = async () => {
    if (onDownload) {
      onDownload()
      return
    }

    // Use Zustand store action instead of local state management
    await startDownload(currentItemId, currentImageUrl, currentItemTitle || 'fashion-item')
  }

  // Early return if no image data
  if (!currentImageUrl) return null

  return (
    <Dialog open={isOpen} onOpenChange={closeZoomModal}>
      <DialogContent className="max-w-[100vw] max-h-[100vh] w-full h-full p-0 bg-black/95">
        {/* 
          Learning: DialogHeader with visually hidden title for accessibility
          Screen readers need a title to understand the dialog purpose,
          but we don't want it visually prominent in a fullscreen image viewer
        */}
        <DialogHeader className="sr-only">
          <DialogTitle>
            {currentItemTitle ? `Image Zoom: ${currentItemTitle}` : 'Image Zoom Viewer'}
          </DialogTitle>
        </DialogHeader>

        {/* Header Controls */}
        <div className="absolute top-0 left-0 right-0 z-50 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {/* Zoom Level Indicator */}
              <Badge variant="secondary" className="bg-black/70 text-white">
                {Math.round(zoom * 100)}%
              </Badge>
              {rotation > 0 && (
                <Badge variant="secondary" className="bg-black/70 text-white">
                  {rotation}°
                </Badge>
              )}
              {/* Download Progress Indicator - Now from Zustand store */}
              {itemIsDownloading && (
                <Badge variant="secondary" className="bg-blue-500 text-white">
                  Downloading {downloadProgress}%
                </Badge>
              )}
              {/* Download Status Indicators */}
              {downloadStatus === 'completed' && (
                <Badge variant="secondary" className="bg-green-500 text-white">
                  Downloaded ✓
                </Badge>
              )}
              {downloadStatus === 'error' && (
                <Badge variant="secondary" className="bg-red-500 text-white">
                  Download Failed ✗
                </Badge>
              )}
            </div>
            
            <DialogClose asChild>
              <Button variant="ghost" size="icon" className="text-white hover:bg-white/20">
                <X className="h-6 w-6" />
                <span className="sr-only">Close image zoom</span>
              </Button>
            </DialogClose>
          </div>
        </div>

        {/* Image Container */}
        <div 
          ref={imageContainerRef}
          className={cn(
            "relative w-full h-full flex items-center justify-center overflow-hidden",
            isDragging ? "cursor-grabbing" : "cursor-grab"
          )}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          onWheel={handleWheel}
        >
          <div
            className="relative transition-transform duration-200 ease-out"
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${zoom}) rotate(${rotation}deg)`,
              transformOrigin: 'center center'
            }}
          >
            <Image
              ref={imageRef}
              src={currentImageUrl}
              alt={currentImageAlt}
              width={800}
              height={600}
              className="max-w-[90vw] max-h-[90vh] object-contain"
              sizes="90vw"
              priority
              /**
               * Learning: High priority loading for zoom modal
               * Users expect immediate image display in fullscreen mode
               */
            />
          </div>
          
          {/* Pan instruction overlay when zoomed */}
          {zoom > 1 && (
            <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 pointer-events-none">
              <Badge variant="secondary" className="bg-black/70 text-white flex items-center gap-1">
                <Move className="h-3 w-3" />
                Drag to pan
              </Badge>
            </div>
          )}
        </div>

        {/* Bottom Controls */}
        <div className="absolute bottom-0 left-0 right-0 z-50 p-4">
          <div className="flex items-center justify-center gap-2">
            {/* Zoom Controls */}
            <Button 
              variant="secondary" 
              size="icon" 
              onClick={zoomOut}
              disabled={zoom <= 0.5}
              className="bg-black/70 text-white hover:bg-black/80"
              title="Zoom out"
              aria-label="Zoom out"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            
            <Button 
              variant="secondary" 
              size="icon" 
              onClick={zoomIn}
              disabled={zoom >= 5}
              className="bg-black/70 text-white hover:bg-black/80"
              title="Zoom in"
              aria-label="Zoom in"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            
            {/* Rotation Control */}
            <Button 
              variant="secondary" 
              size="icon" 
              onClick={rotate}
              className="bg-black/70 text-white hover:bg-black/80"
              title="Rotate 90°"
              aria-label="Rotate image 90 degrees"
            >
              <RotateCw className="h-4 w-4" />
            </Button>
            
            {/* Reset View */}
            <Button 
              variant="secondary" 
              onClick={resetView}
              className="bg-black/70 text-white hover:bg-black/80"
              title="Reset zoom and rotation"
              aria-label="Reset zoom and rotation to default"
            >
              Reset View
            </Button>
            
            {/* Action Buttons */}
            <div className="ml-4 flex gap-2">
              <Button 
                variant="secondary" 
                size="icon" 
                onClick={handleLike}
                className={cn(
                  "bg-black/70 hover:bg-black/80 transition-colors",
                  isItemLiked 
                    ? "text-red-500 hover:text-red-400" 
                    : "text-white hover:text-white"
                )}
                title={isItemLiked ? "Remove from favorites" : "Add to favorites"}
                aria-label={isItemLiked ? "Remove from favorites" : "Add to favorites"}
              >
                <Heart className={cn("h-4 w-4", isItemLiked && "fill-current")} />
              </Button>
              
              <Button 
                variant="secondary" 
                size="icon" 
                onClick={handleDownloadClick}
                disabled={itemIsDownloading}
                className={cn(
                  "bg-black/70 hover:bg-black/80",
                  downloadStatus === 'completed' && "bg-green-600 hover:bg-green-700",
                  downloadStatus === 'error' && "bg-red-600 hover:bg-red-700"
                )}
                title={
                  itemIsDownloading 
                    ? `Downloading ${downloadProgress}%` 
                    : downloadStatus === 'completed'
                    ? "Downloaded successfully"
                    : downloadStatus === 'error'
                    ? "Download failed - click to retry"
                    : "Download image to your computer"
                }
                aria-label={
                  itemIsDownloading 
                    ? `Downloading ${downloadProgress} percent` 
                    : downloadStatus === 'completed'
                    ? "Image downloaded successfully"
                    : downloadStatus === 'error'
                    ? "Download failed, click to retry"
                    : "Download image to your computer"
                }
              >
                {itemIsDownloading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : downloadStatus === 'completed' ? (
                  <Download className="h-4 w-4 text-white" />
                ) : downloadStatus === 'error' ? (
                  <Download className="h-4 w-4 text-white" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Educational Info Overlay */}
        <div className="absolute top-16 right-4 z-40 max-w-xs">
          <div className="bg-black/70 rounded-lg p-3 text-white text-xs space-y-1">
            <p className="font-medium">💡 Zoom Controls:</p>
            <p>• Mouse wheel or buttons to zoom</p>
            <p>• Drag to pan when zoomed in</p>
            <p>• Touch gestures supported on mobile</p>
            <p>• Download state managed by Zustand</p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
} 