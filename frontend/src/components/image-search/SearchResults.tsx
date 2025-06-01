/**
 * @fileoverview Search Results Component - Display grid of semantic search results
 * @learning Demonstrates result rendering, similarity scoring, and responsive grid layouts
 * @concepts Vector similarity visualization, loading states, image optimization, mobile touch UI, Zustand state management
 */

'use client'

import React from 'react'
import Image from 'next/image'
import { Heart, ExternalLink, Download, Eye, Loader2 } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useImageSearchStore, SearchResult } from '@/stores/imageSearchStore'
import { useResultDetailsModal } from '@/stores/modalStore'
import { useLikedItemsStore } from '@/stores/likedItemsStore'
import { useDownloadStore } from '@/stores/downloadStore'
import { ResultDetailsModal } from './ResultDetailsModal'
import { cn } from '@/lib/utils'

interface SearchResultsProps {
  className?: string
  viewMode?: 'grid' | 'list'
}

export const SearchResults = ({ className, viewMode = 'grid' }: SearchResultsProps) => {
  const { results, isSearching, searchMode } = useImageSearchStore()
  
  /**
   * Learning: Using Zustand store for modal state management
   * This demonstrates centralized state management vs local component state
   * Benefits: Better debugging, state persistence, cleaner component logic
   */
  const { isOpen, result, openModal, closeModal } = useResultDetailsModal()

  /**
   * Learning: Handler for opening the details modal using Zustand actions
   * This shows how Zustand actions simplify state management compared to useState
   */
  const handleViewDetails = (searchResult: SearchResult) => {
    openModal(searchResult)
  }

  /**
   * Learning: Early return pattern improves readability and performance
   * We handle different UI states clearly without nested conditionals
   */
  if (isSearching) {
    return <SearchLoadingState className={className} viewMode={viewMode} />
  }

  if (results.length === 0) {
    return <EmptySearchState className={className} />
  }

  return (
    <>
      <div className={cn("w-full", className)}>
        {/* Results Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold">Search Results</h2>
            <p className="text-muted-foreground">
              Found {results.length} similar fashion items
              {searchMode === 'image_to_image' && ' • Sorted by visual similarity'}
            </p>
          </div>
        </div>

        {/* Results Grid/List */}
        <div className={cn(
          // Learning: Dynamic layout based on view mode
          viewMode === 'grid' ? [
            // Grid layout - responsive columns
            "grid gap-6",
            "grid-cols-1",      // Mobile: 1 column
            "sm:grid-cols-2",   // Small screens: 2 columns  
            "lg:grid-cols-3",   // Large screens: 3 columns
            "xl:grid-cols-4"    // Extra large: 4 columns
          ] : [
            // List layout - single column with different card style
            "space-y-4"
          ]
        )}>
          {results.map((result) => (
            <SearchResultCard 
              key={result.id} 
              result={result} 
              viewMode={viewMode}
              onViewDetails={handleViewDetails}
            />
          ))}
        </div>
      </div>

      {/* Educational AI Analysis Modal - Now managed by Zustand */}
      <ResultDetailsModal 
        result={result}
        isOpen={isOpen}
        onClose={closeModal}
      />
    </>
  )
}

/**
 * Individual search result card with similarity scoring and actions
 */
interface SearchResultCardProps {
  result: SearchResult
  viewMode?: 'grid' | 'list'
  onViewDetails: (result: SearchResult) => void
}

const SearchResultCard = ({ result, viewMode = 'grid', onViewDetails }: SearchResultCardProps) => {
  /**
   * Learning: Multiple Zustand stores integration in card component
   * This demonstrates how to use multiple stores for different concerns:
   * - useLikedItemsStore: Favorites functionality
   * - useDownloadStore: Download state and progress tracking
   */
  const { isLiked, toggleLike } = useLikedItemsStore()
  const isItemLiked = isLiked(result.id)

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

  // Get download state for current item using store selectors
  const itemIsDownloading = isDownloading(result.id)
  const downloadProgress = getDownloadProgress(result.id)
  const downloadStatus = getDownloadStatus(result.id)

  /**
   * Learning: Similarity score visualization helps users understand AI confidence
   * Color coding provides immediate visual feedback about result quality
   */
  const getSimilarityColor = (score: number) => {
    if (score >= 0.9) return 'bg-green-500'
    if (score >= 0.8) return 'bg-green-400' 
    if (score >= 0.7) return 'bg-yellow-500'
    if (score >= 0.6) return 'bg-orange-500'
    return 'bg-red-500'
  }

  const getSimilarityLabel = (score: number) => {
    if (score >= 0.9) return 'Excellent match'
    if (score >= 0.8) return 'Very similar'
    if (score >= 0.7) return 'Similar'
    if (score >= 0.6) return 'Somewhat similar'
    return 'Low similarity'
  }

  /**
   * Learning: Action handlers for user interactions
   * Now includes Zustand store actions for both likes and downloads
   */
  const handleViewDetailsClick = () => {
    onViewDetails(result)
  }

  const handleLike = (e: React.MouseEvent) => {
    /**
     * Learning: Prevent event bubbling to avoid triggering parent click handlers
     * This ensures only the like action is triggered, not the card click
     */
    e.stopPropagation()
    
    toggleLike(result.id, {
      imageUrl: result.imageUrl,
      title: result.title
    })
  }

  const handleDownload = async (e: React.MouseEvent) => {
    /**
     * Learning: Enhanced download using Zustand store action
     * All download logic is now centralized in the store with automatic state management
     */
    e.stopPropagation()
    
    // Use Zustand store action instead of local state management
    await startDownload(result.id, result.imageUrl, result.title)
  }

  if (viewMode === 'list') {
    // List view layout - horizontal card
    return (
      <Card className="group hover:shadow-lg transition-all duration-300 overflow-hidden">
        <div className="flex gap-4 p-4">
          {/* Image */}
          <div className="relative w-24 h-24 flex-shrink-0 rounded-lg overflow-hidden bg-muted">
            <Image
              src={result.imageUrl}
              alt={result.description}
              fill
              className="object-cover group-hover:scale-105 transition-transform duration-300"
              sizes="96px"
            />
            
            {/* Similarity Score */}
            <div className="absolute top-1 right-1">
              <Badge 
                className={cn(
                  "text-white font-semibold px-1 py-0 text-xs",
                  getSimilarityColor(result.similarityScore)
                )}
              >
                {Math.round(result.similarityScore * 100)}%
              </Badge>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 space-y-2">
            <div>
              <h3 className="font-semibold text-base line-clamp-1">
                {result.title}
              </h3>
              <p className="text-sm text-muted-foreground line-clamp-2">
                {result.description}
              </p>
            </div>
            
            {/* Similarity Information */}
            <div className="flex items-center gap-4 text-sm">
              <span className="text-muted-foreground">
                {getSimilarityLabel(result.similarityScore)}
              </span>
              <span className="font-medium">
                {Math.round(result.similarityScore * 100)}% match
              </span>
            </div>
            
            {/* Tags */}
            {result.metadata.tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {result.metadata.tags.slice(0, 3).map((tag) => (
                  <Badge key={tag} variant="outline" className="text-xs px-2 py-0">
                    {tag}
                  </Badge>
                ))}
                {result.metadata.tags.length > 3 && (
                  <Badge variant="outline" className="text-xs px-2 py-0">
                    +{result.metadata.tags.length - 3}
                  </Badge>
                )}
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex flex-col gap-2 ml-4">
            <Button 
              size="sm" 
              variant="secondary" 
              onClick={handleViewDetailsClick}
              className="px-3"
            >
              <Eye className="h-4 w-4 mr-1" />
              View
            </Button>
            <div className="flex gap-1">
              <Button 
                size="sm" 
                variant="secondary" 
                onClick={handleLike}
                className={cn(
                  "px-2 transition-colors duration-200",
                  isItemLiked 
                    ? "text-red-500 hover:text-red-600" 
                    : "hover:text-red-500"
                )}
                title={isItemLiked ? "Remove from favorites" : "Add to favorites"}
              >
                <Heart className={cn("h-4 w-4", isItemLiked && "fill-current")} />
              </Button>
              <Button 
                size="sm" 
                variant="secondary" 
                onClick={handleDownload}
                disabled={itemIsDownloading}
                className={cn(
                  "px-2",
                  downloadStatus === 'completed' && "text-green-600",
                  downloadStatus === 'error' && "text-red-600"
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
              >
                {itemIsDownloading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </Card>
    )
  }

  // Grid view layout - improved for mobile with no top white space
  return (
    <Card className="group hover:shadow-lg transition-all duration-300 overflow-hidden p-0">
      {/* 
        Learning: Removed CardHeader and added p-0 to Card to eliminate all white space
        The image container now starts at the absolute top edge of the card
      */}
      <div className="relative">
        {/* Image Container - Starts at absolute top with no padding/margin */}
        <div className="relative aspect-[3/4] overflow-hidden bg-muted rounded-t-lg">
          <Image
            src={result.imageUrl}
            alt={result.description}
            fill
            className="object-cover group-hover:scale-105 transition-transform duration-300"
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
            /**
             * Learning: Next.js Image optimization provides:
             * - Automatic WebP/AVIF conversion
             * - Responsive image sizing
             * - Lazy loading by default
             * - Layout shift prevention
             */
          />
          
          {/* Similarity Score Overlay */}
          <div className="absolute top-3 right-3">
            <Badge 
              className={cn(
                "text-white font-semibold px-2 py-1",
                getSimilarityColor(result.similarityScore)
              )}
            >
              {Math.round(result.similarityScore * 100)}%
            </Badge>
          </div>

          {/* Like Status Indicator */}
          {isItemLiked && (
            <div className="absolute top-3 left-3">
              <Badge className="bg-red-500 text-white px-2 py-1">
                <Heart className="h-3 w-3 fill-current mr-1" />
                Liked
              </Badge>
            </div>
          )}

          {/* Download Status Indicators - Now from Zustand store */}
          {itemIsDownloading && (
            <div className="absolute bottom-3 left-3">
              <Badge className="bg-blue-500 text-white px-2 py-1">
                <Loader2 className="h-3 w-3 animate-spin mr-1" />
                {downloadProgress}%
              </Badge>
            </div>
          )}
          {downloadStatus === 'completed' && (
            <div className="absolute bottom-3 left-3">
              <Badge className="bg-green-500 text-white px-2 py-1">
                Downloaded ✓
              </Badge>
            </div>
          )}
          {downloadStatus === 'error' && (
            <div className="absolute bottom-3 left-3">
              <Badge className="bg-red-500 text-white px-2 py-1">
                Failed ✗
              </Badge>
            </div>
          )}

          {/* Action Buttons Overlay */}
          <div className={cn(
            "absolute inset-0 transition-colors duration-300",
            // Learning: Different interaction patterns for touch vs mouse
            // Mobile: Always show slight background for better contrast
            // Desktop: Show background on hover
            "md:bg-black/0 md:group-hover:bg-black/20"
          )}>
            <div className={cn(
              "absolute bottom-3 left-3 right-3 transition-opacity duration-300",
              // Learning: Always show buttons on mobile (< md), hover on desktop (>= md)
              "opacity-100", // Always visible on mobile
              "md:opacity-0 md:group-hover:opacity-100" // Hover to show on desktop only
            )}>
              <div className="flex gap-2">
                <Button 
                  size="sm" 
                  variant="secondary" 
                  onClick={handleViewDetailsClick}
                  className={cn(
                    "flex-1 backdrop-blur-sm",
                    // Learning: Better contrast on mobile, inherit desktop styling
                    "bg-white/90 hover:bg-white border shadow-sm",
                    "md:bg-white/80 md:hover:bg-white/90"
                  )}
                >
                  <Eye className="h-4 w-4 mr-1" />
                  View
                </Button>
                <Button 
                  size="sm" 
                  variant="secondary" 
                  onClick={handleLike}
                  className={cn(
                    "backdrop-blur-sm transition-colors duration-200",
                    "bg-white/90 hover:bg-white border shadow-sm",
                    "md:bg-white/80 md:hover:bg-white/90",
                    isItemLiked 
                      ? "text-red-500 hover:text-red-600" 
                      : "hover:text-red-500"
                  )}
                  title={isItemLiked ? "Remove from favorites" : "Add to favorites"}
                >
                  <Heart className={cn("h-4 w-4", isItemLiked && "fill-current")} />
                </Button>
                <Button 
                  size="sm" 
                  variant="secondary" 
                  onClick={handleDownload}
                  disabled={itemIsDownloading}
                  className={cn(
                    "backdrop-blur-sm",
                    "bg-white/90 hover:bg-white border shadow-sm",
                    "md:bg-white/80 md:hover:bg-white/90",
                    downloadStatus === 'completed' && "text-green-600",
                    downloadStatus === 'error' && "text-red-600"
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
                >
                  {itemIsDownloading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <CardContent className="p-4 space-y-3">
        {/* Title and Description */}
        <div className="space-y-1">
          <h3 className="font-semibold text-sm line-clamp-1">
            {result.title}
          </h3>
          <p className="text-xs text-muted-foreground line-clamp-2">
            {result.description}
          </p>
        </div>

        {/* Similarity Information */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">
            {getSimilarityLabel(result.similarityScore)}
          </span>
          <span className="font-medium">
            {Math.round(result.similarityScore * 100)}% match
          </span>
        </div>

        {/* Tags */}
        {result.metadata.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {result.metadata.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs px-2 py-0">
                {tag}
              </Badge>
            ))}
            {result.metadata.tags.length > 3 && (
              <Badge variant="outline" className="text-xs px-2 py-0">
                +{result.metadata.tags.length - 3}
              </Badge>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

/**
 * Loading state with skeleton cards for better perceived performance
 */
interface LoadingStateProps {
  className?: string
  viewMode?: 'grid' | 'list'
}

const SearchLoadingState = ({ className, viewMode = 'grid' }: LoadingStateProps) => (
  <div className={cn("w-full", className)}>
    <div className="mb-6">
      <Skeleton className="h-8 w-48 mb-2" />
      <Skeleton className="h-4 w-64" />
    </div>
    
    <div className={cn(
      // Learning: Dynamic layout based on view mode
      viewMode === 'grid' ? [
        // Grid layout - responsive columns
        "grid gap-6",
        "grid-cols-1",      // Mobile: 1 column
        "sm:grid-cols-2",   // Small screens: 2 columns  
        "lg:grid-cols-3",   // Large screens: 3 columns
        "xl:grid-cols-4"    // Extra large: 4 columns
      ] : [
        // List layout - single column with different card style
        "space-y-4"
      ]
    )}>
      {Array.from({ length: 8 }).map((_, i) => (
        <Card key={i} className="overflow-hidden">
          {viewMode === 'grid' ? (
            // Grid card layout
            <>
              <Skeleton className="aspect-[3/4] w-full" />
              <CardContent className="p-4 space-y-3">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-2/3" />
                <div className="flex gap-1">
                  <Skeleton className="h-5 w-16" />
                  <Skeleton className="h-5 w-12" />
                  <Skeleton className="h-5 w-14" />
                </div>
              </CardContent>
            </>
          ) : (
            // List card layout
            <div className="flex gap-4 p-4">
              <Skeleton className="w-24 h-24 rounded-lg flex-shrink-0" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-5 w-3/4" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-2/3" />
                <div className="flex gap-2">
                  <Skeleton className="h-6 w-16" />
                  <Skeleton className="h-6 w-20" />
                </div>
              </div>
            </div>
          )}
        </Card>
      ))}
    </div>
  </div>
)

/**
 * Empty state when no search has been performed or no results found
 */
const EmptySearchState = ({ className }: LoadingStateProps) => (
  <div className={cn(
    "flex flex-col items-center justify-center py-16 text-center",
    className
  )}>
    <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center mb-6">
      <ExternalLink className="h-12 w-12 text-muted-foreground" />
    </div>
    
    <h3 className="text-xl font-semibold mb-2">No results yet</h3>
    <p className="text-muted-foreground max-w-md">
      Start by searching for fashion items using text descriptions or upload an image to find similar items.
    </p>
  </div>
) 