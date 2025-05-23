/**
 * @fileoverview Result Details Modal - Educational AI transparency and analysis viewer
 * @learning Demonstrates vector similarity analysis, AI confidence scoring, and responsive modal patterns
 * @concepts Vector embeddings visualization, similarity explanations, responsive UI with Dialog/Drawer, Zustand integration
 */

'use client'

import React from 'react'
import Image from 'next/image'
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle,
  DialogDescription 
} from '@/components/ui/dialog'
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerDescription,
  DrawerFooter,
  DrawerClose
} from '@/components/ui/drawer'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Eye, 
  Heart, 
  Download, 
  ZoomIn, 
  Shirt, 
  Brain,
  TrendingUp,
  Info,
  Loader2
} from 'lucide-react'
import { useMediaQuery } from '@/hooks/useMediaQuery'
import { SearchResult } from '@/stores/imageSearchStore'
import { useResultDetailsModal } from '@/stores/modalStore'
import { useLikedItemsStore } from '@/stores/likedItemsStore'
import { useImageZoomStore } from '@/stores/imageZoomStore'
import { useDownloadStore } from '@/stores/downloadStore'
import { ImageZoomModal } from './ImageZoomModal'
import { cn } from '@/lib/utils'

interface ResultDetailsModalProps {
  result: SearchResult | null
  isOpen: boolean
  onClose: () => void
}

// TypeScript interfaces for AI analysis data
interface FeatureBreakdown {
  color: number
  style: number
  pattern: number
  silhouette: number
  texture: number
}

interface DetectedFeature {
  name: string
  value: string
  confidence: number
}

interface VectorDimension {
  dimension: number
  value: number
  feature: string
}

interface AIAnalysisData {
  overallConfidence: number
  featureBreakdown: FeatureBreakdown
  detectedFeatures: DetectedFeature[]
  vectorDimensions: number
  matchingReasons: string[]
  embeddingVisualization: {
    topDimensions: VectorDimension[]
  }
}

/**
 * Learning: Responsive modal pattern using Dialog for desktop, Drawer for mobile
 * This pattern provides optimal UX for different screen sizes and interaction methods
 * Enhanced with multiple Zustand stores for centralized state management
 */
export const ResultDetailsModal = ({ result, isOpen, onClose }: ResultDetailsModalProps) => {
  const isDesktop = useMediaQuery('(min-width: 768px)')
  
  /**
   * Learning: Using multiple Zustand stores for different concerns
   * - useResultDetailsModal: Tab state management
   * - useLikedItemsStore: Favorites functionality  
   * - useImageZoomStore: Zoom modal functionality
   * - useDownloadStore: Download state and progress tracking
   * This demonstrates how to compose multiple stores in a single component
   */
  const { activeTab, setActiveTab } = useResultDetailsModal()
  const { isLiked, toggleLike } = useLikedItemsStore()
  const { openZoomModal } = useImageZoomStore()

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

  if (!result) return null

  // Check if current item is liked
  const isItemLiked = isLiked(result.id)

  // Get download state for current item using store selectors
  const itemIsDownloading = isDownloading(result.id)
  const downloadProgress = getDownloadProgress(result.id)
  const downloadStatus = getDownloadStatus(result.id)

  /**
   * Learning: Action handlers using Zustand store actions
   * All handlers use store actions instead of local state management
   */
  const handleZoom = () => {
    /**
     * Learning: Opening zoom modal through Zustand store action
     * This eliminates the need for local state and prop passing
     */
    openZoomModal(
      result.imageUrl,
      result.description,
      result.id,
      result.title
    )
  }

  const handleLike = () => {
    /**
     * Learning: Optimistic UI updates for better perceived performance
     * Zustand handles the state update immediately for responsive UI
     */
    toggleLike(result.id, {
      imageUrl: result.imageUrl,
      title: result.title
    })
  }

  const handleDownload = async () => {
    /**
     * Learning: Enhanced download using Zustand store action
     * All download logic is now centralized in the store with automatic state management
     */
    await startDownload(result.id, result.imageUrl, result.title)
  }

  /**
   * Learning: Mock AI analysis data for educational purposes
   * In production, this would come from your AI service with real vector analysis
   */
  const mockAIAnalysis: AIAnalysisData = {
    overallConfidence: result.similarityScore,
    featureBreakdown: {
      color: 0.95,
      style: 0.87,
      pattern: 0.92,
      silhouette: 0.84,
      texture: 0.78
    },
    detectedFeatures: [
      { name: 'Collar Type', value: 'Round Neck', confidence: 0.94 },
      { name: 'Sleeve Length', value: 'Short Sleeve', confidence: 0.92 },
      { name: 'Pattern', value: 'Solid Color', confidence: 0.89 },
      { name: 'Fit', value: 'Regular Fit', confidence: 0.87 }
    ],
    vectorDimensions: 768,
    matchingReasons: [
      'Similar color palette (blue tones)',
      'Matching garment silhouette',
      'Comparable fabric texture patterns',
      'Similar style classification'
    ],
    embeddingVisualization: {
      // Mock vector data for educational display
      topDimensions: [
        { dimension: 42, value: 0.8234, feature: 'Color saturation' },
        { dimension: 156, value: 0.7892, feature: 'Garment shape' },
        { dimension: 334, value: 0.7445, feature: 'Texture pattern' },
        { dimension: 501, value: 0.6789, feature: 'Style category' }
      ]
    }
  }

  const ModalContent = () => (
    <div className="space-y-6">
      {/* Header with Image and Basic Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Product Image */}
        <div className="space-y-4">
          <div className="relative aspect-[3/4] rounded-lg overflow-hidden bg-muted">
            <Image
              src={result.imageUrl}
              alt={result.description}
              fill
              className="object-cover"
              sizes="(max-width: 768px) 100vw, 50vw"
            />
            {/* Download Progress Overlay - Now from Zustand store */}
            {itemIsDownloading && (
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                <div className="bg-white rounded-lg p-4 text-center">
                  <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2" />
                  <p className="text-sm font-medium">Downloading {downloadProgress}%</p>
                </div>
              </div>
            )}
          </div>
          
          {/* Image Actions - Enhanced with download progress */}
          <div className="flex gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              className="flex-1"
              onClick={handleZoom}
              disabled={itemIsDownloading}
            >
              <ZoomIn className="h-4 w-4 mr-2" />
              Zoom
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleLike}
              disabled={itemIsDownloading}
              className={cn(
                "transition-colors duration-200",
                isItemLiked 
                  ? "text-red-500 border-red-500 hover:bg-red-50 hover:text-red-600" 
                  : "hover:text-red-500 hover:border-red-500"
              )}
              title={isItemLiked ? "Remove from favorites" : "Add to favorites"}
            >
              <Heart className={cn("h-4 w-4", isItemLiked && "fill-current")} />
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleDownload}
              disabled={itemIsDownloading}
              className={cn(
                downloadStatus === 'completed' && "text-green-600 border-green-600",
                downloadStatus === 'error' && "text-red-600 border-red-600"
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

        {/* Product Details */}
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold">{result.title}</h2>
            <p className="text-muted-foreground mt-1">{result.description}</p>
          </div>

          {/* Similarity Score */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Overall Match</span>
                <Badge className="bg-green-500">
                  {Math.round(mockAIAnalysis.overallConfidence * 100)}%
                </Badge>
              </div>
              <Progress 
                value={mockAIAnalysis.overallConfidence * 100} 
                className="h-2"
              />
              <p className="text-xs text-muted-foreground mt-2">
                This item matches your search criteria with high confidence
              </p>
            </CardContent>
          </Card>

          {/* Like Status Indicator */}
          {isItemLiked && (
            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Heart className="h-4 w-4 text-red-500 fill-current" />
                  <span className="text-sm font-medium text-red-700">
                    Added to your favorites
                  </span>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Download Status Indicators - Now from Zustand store */}
          {itemIsDownloading && (
            <Card className="border-blue-200 bg-blue-50">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
                  <span className="text-sm font-medium text-blue-700">
                    Downloading image... {downloadProgress}%
                  </span>
                </div>
                <Progress value={downloadProgress} className="h-2 mt-2" />
              </CardContent>
            </Card>
          )}
          {downloadStatus === 'completed' && (
            <Card className="border-green-200 bg-green-50">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium text-green-700">
                    Image downloaded successfully
                  </span>
                </div>
              </CardContent>
            </Card>
          )}
          {downloadStatus === 'error' && (
            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-red-500" />
                  <span className="text-sm font-medium text-red-700">
                    Download failed - click to retry
                  </span>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Tags */}
          {result.metadata.tags.length > 0 && (
            <div>
              <h4 className="font-medium mb-2">Tags</h4>
              <div className="flex flex-wrap gap-2">
                {result.metadata.tags.map((tag) => (
                  <Badge key={tag} variant="outline">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* AI Analysis Tabs - Controlled by Zustand store */}
      <Tabs 
        value={activeTab} 
        onValueChange={(value) => {
          // Type-safe tab switching with Zustand store
          if (value === 'analysis' || value === 'features' || value === 'vectors') {
            setActiveTab(value)
          }
        }} 
        className="w-full"
      >
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="analysis">
            <Brain className="h-4 w-4 mr-2" />
            AI Analysis
          </TabsTrigger>
          <TabsTrigger value="features">
            <Eye className="h-4 w-4 mr-2" />
            Features
          </TabsTrigger>
          <TabsTrigger value="vectors">
            <TrendingUp className="h-4 w-4 mr-2" />
            Vector Data
          </TabsTrigger>
        </TabsList>

        <TabsContent value="analysis" className="space-y-4 mt-6">
          <AIAnalysisTab analysis={mockAIAnalysis} />
        </TabsContent>

        <TabsContent value="features" className="space-y-4 mt-6">
          <FeaturesTab analysis={mockAIAnalysis} />
        </TabsContent>

        <TabsContent value="vectors" className="space-y-4 mt-6">
          <VectorAnalysisTab analysis={mockAIAnalysis} />
        </TabsContent>
      </Tabs>
    </div>
  )

  if (isDesktop) {
    return (
      <>
        <Dialog open={isOpen} onOpenChange={onClose}>
          <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Product Analysis</DialogTitle>
              <DialogDescription>
                Detailed AI analysis and vector similarity breakdown
              </DialogDescription>
            </DialogHeader>
            <ModalContent />
          </DialogContent>
        </Dialog>
        
        {/* Zoom Modal - Now managed entirely by Zustand store */}
        <ImageZoomModal onDownload={handleDownload} />
      </>
    )
  }

  return (
    <>
      <Drawer open={isOpen} onOpenChange={onClose}>
        <DrawerContent className="max-h-[90vh]">
          <DrawerHeader>
            <DrawerTitle>Product Analysis</DrawerTitle>
            <DrawerDescription>
              Detailed AI analysis and vector similarity breakdown
            </DrawerDescription>
          </DrawerHeader>
          <div className="px-4 pb-6 overflow-y-auto">
            <ModalContent />
          </div>
          <DrawerFooter>
            <DrawerClose asChild>
              <Button variant="outline">Close</Button>
            </DrawerClose>
          </DrawerFooter>
        </DrawerContent>
      </Drawer>
      
      {/* Zoom Modal - Now managed entirely by Zustand store */}
      <ImageZoomModal onDownload={handleDownload} />
    </>
  )
}

/**
 * AI Analysis Tab - Shows confidence breakdown and matching reasons
 */
const AIAnalysisTab = ({ analysis }: { analysis: AIAnalysisData }) => (
  <div className="space-y-6">
    {/* Feature Confidence Breakdown */}
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Feature Confidence Breakdown
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/**
         * Learning: Feature confidence shows how AI models analyze different aspects
         * Each feature represents a different aspect the neural network evaluates
         */}
        {Object.entries(analysis.featureBreakdown).map(([feature, confidence]) => (
          <div key={feature} className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="capitalize">{feature}</span>
              <span className="font-medium">{Math.round(confidence * 100)}%</span>
            </div>
            <Progress value={confidence * 100} className="h-2" />
          </div>
        ))}
        
        <div className="pt-4 border-t">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-muted-foreground">
              <strong>Learning:</strong> These scores show how confident the AI model is about 
              different visual features. Higher scores indicate stronger similarity in that aspect.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>

    {/* Why This Matched */}
    <Card>
      <CardHeader>
        <CardTitle>Why This Matched Your Search</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {analysis.matchingReasons.map((reason: string, index: number) => (
            <li key={index} className="flex items-start gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 mt-2 flex-shrink-0" />
              <span className="text-sm">{reason}</span>
            </li>
          ))}
        </ul>
        
        <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-blue-900">How Vector Similarity Works</p>
              <p className="text-xs text-blue-700 mt-1">
                The AI converts images into numerical vectors (embeddings) that capture semantic meaning. 
                Similar images have similar vector patterns, allowing for semantic search beyond just keywords.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  </div>
)

/**
 * Features Tab - Shows detected visual features
 */
const FeaturesTab = ({ analysis }: { analysis: AIAnalysisData }) => (
  <div className="space-y-6">
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-5 w-5" />
          Detected Features
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {analysis.detectedFeatures.map((feature: DetectedFeature, index: number) => (
            <div key={index} className="p-3 border rounded-lg">
              <div className="flex justify-between items-start mb-2">
                <span className="font-medium text-sm">{feature.name}</span>
                <Badge variant="outline" className="text-xs">
                  {Math.round(feature.confidence * 100)}%
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">{feature.value}</p>
              <Progress value={feature.confidence * 100} className="h-1 mt-2" />
            </div>
          ))}
        </div>

        <div className="mt-6 p-4 bg-purple-50 rounded-lg border border-purple-200">
          <div className="flex items-start gap-2">
            <Shirt className="h-4 w-4 text-purple-600 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-purple-900">Computer Vision at Work</p>
              <p className="text-xs text-purple-700 mt-1">
                These features are automatically detected using convolutional neural networks 
                trained on millions of fashion images. Each detection includes a confidence score.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  </div>
)

/**
 * Vector Analysis Tab - Educational display of embedding data
 */
const VectorAnalysisTab = ({ analysis }: { analysis: AIAnalysisData }) => (
  <div className="space-y-6">
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Vector Embedding Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-center">
          <div className="p-4 bg-muted rounded-lg">
            <div className="text-2xl font-bold">{analysis.vectorDimensions}</div>
            <div className="text-sm text-muted-foreground">Vector Dimensions</div>
          </div>
          <div className="p-4 bg-muted rounded-lg">
            <div className="text-2xl font-bold">
              {Math.round(analysis.overallConfidence * 100)}%
            </div>
            <div className="text-sm text-muted-foreground">Similarity Score</div>
          </div>
        </div>

        {/* Top Contributing Dimensions */}
        <div>
          <h4 className="font-medium mb-3">Top Contributing Vector Dimensions</h4>
          <div className="space-y-3">
            {analysis.embeddingVisualization.topDimensions.map((dim: VectorDimension, index: number) => (
              <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <div className="font-medium text-sm">Dimension {dim.dimension}</div>
                  <div className="text-xs text-muted-foreground">{dim.feature}</div>
                </div>
                <div className="text-right">
                  <div className="font-medium">{dim.value.toFixed(4)}</div>
                  <div className="text-xs text-muted-foreground">Activation</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-4 bg-green-50 rounded-lg border border-green-200">
          <div className="flex items-start gap-2">
            <TrendingUp className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-green-900">Vector Embeddings Explained</p>
              <p className="text-xs text-green-700 mt-1">
                Neural networks convert images into high-dimensional vectors where similar images 
                cluster together. These numbers represent learned visual concepts that enable 
                semantic similarity search.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  </div>
) 