/**
 * @fileoverview Search results display component
 * @learning Demonstrates rendering API data with performance optimization
 * @concepts Virtual scrolling, responsive grids, lazy loading, similarity visualization
 */

'use client'

import Image from 'next/image'
import { ShoppingBag, TrendingUp, Calendar, Palette, Tag, Eye } from 'lucide-react'
import { Card, CardContent, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Progress } from '@/components/ui/progress'
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle,
  DialogDescription,
  DialogClose
} from '@/components/ui/dialog'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'
import { useProductDetailsModal } from '@/stores/modalStore'
import type { SearchResponse, FashionProduct } from '@/types/api'

interface SearchResultsProps {
  results: SearchResponse | null
  isLoading?: boolean
  className?: string
}

/**
 * Main search results component with enhanced dialog functionality
 * @learning Demonstrates Zustand state management for modal interactions
 */
export function SearchResults({ results, isLoading, className }: SearchResultsProps) {
  /**
   * Learning: Using Zustand store for modal state management
   * This replaces useState with centralized state management
   * Benefits: Better debugging, state persistence, cleaner component logic
   */
  const { isOpen, product, openModal, closeModal } = useProductDetailsModal()

  // Calculate similarity score ranges for color coding
  /**
   * Learning: Color-coded similarity scores provide instant visual feedback
   * Users can quickly identify best matches without reading exact percentages
   */
  const getScoreColor = (score: number): string => {
    if (score >= 0.9) return 'bg-green-500'
    if (score >= 0.8) return 'bg-green-400'
    if (score >= 0.7) return 'bg-yellow-500'
    if (score >= 0.6) return 'bg-orange-500'
    return 'bg-red-500'
  }

  const formatScore = (score: number): string => {
    return `${(score * 100).toFixed(1)}%`
  }

  /**
   * Handle opening the product details dialog using Zustand action
   * @learning Zustand actions simplify state management compared to useState
   */
  const handleViewDetails = (product: FashionProduct) => {
    openModal(product)
  }

  /**
   * Learning: Handle loading state and null results during search transitions
   * This prevents showing stale data when a new search starts
   */
  if (isLoading || !results) {
    return (
      <div className={cn('space-y-6', className)}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {Array.from({ length: 8 }).map((_, i) => (
            <Card key={i} className="overflow-hidden">
              <Skeleton className="aspect-[3/4] w-full" />
              <CardContent className="p-4 space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className={cn('space-y-6', className)}>
      {/* Results Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">
            {results.query_type === 'text-to-image' ? 'Text Search' : 'Visual Search'} Results
          </h2>
          <p className="text-muted-foreground mt-1">
            Found {results.results_count} similar products
            {results.performance && ` in ${results.performance.total_time}s`}
          </p>
        </div>

        {/* Performance Metrics */}
        {results.performance && (
          <div className="text-sm text-muted-foreground space-y-1">
            <div>Embedding: {results.performance.embedding_generation}s</div>
            <div>Search: {results.performance.vector_search}s</div>
          </div>
        )}
      </div>

      {/* Similarity Statistics */}
      {results.similarity_stats && (
        <Card className="p-4 bg-muted/50">
          <h3 className="font-semibold mb-3">Similarity Distribution</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Highest:</span>
              <p className="font-semibold">{formatScore(results.similarity_stats.max)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Average:</span>
              <p className="font-semibold">{formatScore(results.similarity_stats.average)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Lowest:</span>
              <p className="font-semibold">{formatScore(results.similarity_stats.min)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Std Dev:</span>
              <p className="font-semibold">{formatScore(results.similarity_stats.std_dev)}</p>
            </div>
          </div>
        </Card>
      )}

      {/* Results Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {results.results.map((product) => (
          <ProductCard
            key={product.id}
            product={product}
            scoreColor={getScoreColor(product.similarity_score)}
            onViewDetails={handleViewDetails}
          />
        ))}
      </div>

      {/* No Results State */}
      {results.results_count === 0 && (
        <Card className="p-12 text-center">
          <ShoppingBag className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Results Found</h3>
          <p className="text-muted-foreground">
            Try adjusting your search query or uploading a different image
          </p>
        </Card>
      )}

      {/* Product Details Dialog - Now managed entirely by Zustand store */}
      <ProductDetailsDialog
        product={product}
        isOpen={isOpen}
        onClose={closeModal}
      />
    </div>
  )
}

/**
 * Individual product card component
 * @learning Component composition for maintainability
 */
function ProductCard({ 
  product, 
  scoreColor,
  onViewDetails
}: { 
  product: FashionProduct
  scoreColor: string
  onViewDetails: (product: FashionProduct) => void
}) {
  return (
    <Card className="overflow-hidden group hover:shadow-lg transition-shadow">
      {/* Product Image */}
      <div className="relative aspect-[3/4] bg-gray-100">
        <Image
          src={product.image}
          alt={product.name}
          fill
          className="object-cover group-hover:scale-105 transition-transform duration-300"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
        />
        
        {/* Similarity Score Badge */}
        <div className="absolute top-2 right-2">
          <Badge className={cn('text-white border-0', scoreColor)}>
            {Math.round(product.similarity_score * 100)}% Match
          </Badge>
        </div>

        {/* Category Badge */}
        {product.master_category && (
          <Badge 
            variant="secondary" 
            className="absolute top-2 left-2 bg-black/70 text-white"
          >
            {product.master_category}
          </Badge>
        )}
      </div>

      <CardContent className="p-4 space-y-3">
        {/* Product Name & Brand */}
        <div>
          <h3 className="font-semibold line-clamp-1">{product.name}</h3>
          {product.brand && (
            <p className="text-sm text-muted-foreground">{product.brand}</p>
          )}
        </div>

        {/* Product Attributes */}
        <div className="flex flex-wrap gap-2">
          {product.gender && (
            <Badge variant="outline" className="text-xs">
              {product.gender}
            </Badge>
          )}
          
          {product.base_colour && (
            <Badge variant="outline" className="text-xs">
              <Palette className="h-3 w-3 mr-1" />
              {product.base_colour}
            </Badge>
          )}
          
          {product.season && (
            <Badge variant="outline" className="text-xs">
              <Calendar className="h-3 w-3 mr-1" />
              {product.season}
            </Badge>
          )}
        </div>

        {/* Additional Metadata */}
        <div className="space-y-1 text-xs text-muted-foreground">
          {product.article_type && (
            <div className="flex items-center gap-1">
              <Tag className="h-3 w-3" />
              <span>{product.article_type}</span>
            </div>
          )}
          
          {product.usage && (
            <div className="flex items-center gap-1">
              <TrendingUp className="h-3 w-3" />
              <span>{product.usage}</span>
            </div>
          )}
        </div>

        {/* Similarity Progress Bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Similarity</span>
            <span className="font-medium">{Math.round(product.similarity_score * 100)}%</span>
          </div>
          <Progress 
            value={product.similarity_score * 100} 
            className="h-1.5"
          />
        </div>
      </CardContent>

      <CardFooter className="p-4 pt-0">
        <Button 
          variant="outline" 
          size="sm" 
          className="w-full"
          onClick={() => onViewDetails(product)}
        >
          <Eye className="h-4 w-4 mr-2" />
          View Details
        </Button>
      </CardFooter>
    </Card>
  )
}

/**
 * Format product description HTML content for better rendering
 * @learning Demonstrates HTML content processing for better UX
 */
function formatDescriptionContent(htmlContent: string): string {
  let formatted = htmlContent
  
  // First, clean up any existing HTML to ensure consistent formatting
  formatted = formatted.replace(/<\/?p[^>]*>/g, '')
  
  // Handle numbered points at the start of lines (1., 2., 3., etc.)
  formatted = formatted.replace(
    /^(\d+\.\s+[^\n\r]*?)$/gm,
    '<p class="font-medium text-gray-800 mb-2 pl-2">$1</p>'
  )
  
  // Handle section headers ending with colon (like "Model's Statistics:")
  formatted = formatted.replace(
    /^([A-Z][^:\n\r]*:)\s*$/gm,
    '<p class="font-semibold text-gray-900 mt-6 mb-3 text-base">$1</p>'
  )
  
  // Handle regular paragraphs (lines that don't start with numbers or end with colon)
  formatted = formatted.replace(
    /^(?!\d+\.)(?![A-Z][^:\n\r]*:$)([^\n\r<]+)$/gm,
    (match) => {
      const trimmed = match.trim()
      if (trimmed.length > 0) {
        return `<p class="mb-3 text-gray-700 leading-relaxed">${trimmed}</p>`
      }
      return ''
    }
  )
  
  // Clean up multiple line breaks
  formatted = formatted.replace(/<br\s*\/?>\s*<br\s*\/?>/g, '<br/>')
  formatted = formatted.replace(/(<br\s*\/?>){3,}/g, '<br/><br/>')
  
  // Remove empty paragraphs
  formatted = formatted.replace(/<p[^>]*>\s*<\/p>/g, '')
  
  // Ensure proper spacing between sections
  formatted = formatted.replace(
    /(<\/p>)\s*(<p class="font-semibold)/g,
    '$1<div class="mt-4"></div>$2'
  )
  
  return formatted
}

/**
 * Comprehensive product details dialog component
 * @learning Modal design for detailed data presentation and user engagement
 * @concepts Data visualization, responsive design, accessibility, HTML content rendering
 */
function ProductDetailsDialog({ 
  product, 
  isOpen, 
  onClose 
}: { 
  product: FashionProduct | null
  isOpen: boolean
  onClose: () => void
}) {
  if (!product) return null

  /**
   * Generate similarity insights for educational purposes
   * @learning Explaining AI similarity scoring to users
   */
  const getSimilarityInsight = (score: number) => {
    if (score >= 0.9) return {
      level: "Excellent Match",
      description: "This item has very high visual similarity to your search criteria.",
      color: "text-green-600",
      bgColor: "bg-green-50"
    }
    if (score >= 0.8) return {
      level: "Great Match", 
      description: "This item closely matches your search with strong visual similarity.",
      color: "text-green-600",
      bgColor: "bg-green-50"
    }
    if (score >= 0.7) return {
      level: "Good Match",
      description: "This item has good visual similarity to your search criteria.",
      color: "text-yellow-600", 
      bgColor: "bg-yellow-50"
    }
    if (score >= 0.6) return {
      level: "Moderate Match",
      description: "This item has some visual similarity but may be more exploratory.",
      color: "text-orange-600",
      bgColor: "bg-orange-50"
    }
    return {
      level: "Low Match",
      description: "This item has limited visual similarity - consider it for inspiration.",
      color: "text-red-600",
      bgColor: "bg-red-50"
    }
  }

  const insight = getSimilarityInsight(product.similarity_score)

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[95vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Product Details
          </DialogTitle>
          <DialogDescription>
            Comprehensive product information and AI similarity analysis
          </DialogDescription>
        </DialogHeader>

        <div className="overflow-y-auto max-h-[calc(95vh-120px)] pr-2">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            {/* Product Image Section */}
            <div className="space-y-4">
              <div className="relative aspect-[3/4] rounded-lg overflow-hidden bg-muted">
                <Image
                  src={product.image}
                  alt={product.name}
                  fill
                  className="object-cover"
                  sizes="(max-width: 1024px) 100vw, 50vw"
                />
              </div>
              
              {/* Similarity Analysis Card */}
              <Card className={cn("p-4", insight.bgColor)}>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold">AI Similarity Analysis</h4>
                  <Badge className={cn("text-white", 
                    product.similarity_score >= 0.8 ? "bg-green-500" :
                    product.similarity_score >= 0.7 ? "bg-yellow-500" :
                    product.similarity_score >= 0.6 ? "bg-orange-500" : "bg-red-500"
                  )}>
                    {Math.round(product.similarity_score * 100)}%
                  </Badge>
                </div>
                <div className="space-y-2">
                  <p className={cn("font-medium", insight.color)}>{insight.level}</p>
                  <p className="text-sm text-muted-foreground">{insight.description}</p>
                  <Progress value={product.similarity_score * 100} className="h-2" />
                </div>
              </Card>
            </div>

            {/* Product Information Section */}
            <div className="space-y-6">
              {/* Basic Information */}
              <div>
                <h2 className="text-2xl font-bold mb-2">{product.name}</h2>
                {product.brand && (
                  <p className="text-lg text-muted-foreground mb-4">{product.brand}</p>
                )}
              </div>

              {/* Product Description - Enhanced HTML rendering for structured content */}
              {product.description && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Description</h3>
                  <Card className="p-4 bg-muted/30 max-h-80 overflow-y-auto">
                    <div 
                      className={cn(
                        "text-sm leading-relaxed",
                        // Paragraph styling with better spacing
                        "[&>p]:mb-3 [&>p]:leading-relaxed",
                        // Numbered points styling
                        "[&>p.font-medium]:bg-blue-50 [&>p.font-medium]:border-l-4 [&>p.font-medium]:border-blue-300 [&>p.font-medium]:py-2 [&>p.font-medium]:px-3 [&>p.font-medium]:rounded-r",
                        // Section headers styling
                        "[&>p.font-semibold]:border-b [&>p.font-semibold]:border-gray-200 [&>p.font-semibold]:pb-2",
                        // Strong/bold text styling
                        "[&>strong]:font-semibold [&>strong]:text-gray-900",
                        // List styling - both ordered and unordered
                        "[&>ul]:list-disc [&>ul]:pl-6 [&>ul]:my-4 [&>ul]:space-y-2",
                        "[&>ol]:list-decimal [&>ol]:pl-6 [&>ol]:my-4 [&>ol]:space-y-2",
                        "[&>li]:text-gray-700 [&>li]:leading-relaxed",
                        // Line break handling
                        "[&>br]:block [&>br]:mb-2",
                        // Spacing improvements
                        "[&>div]:mb-2"
                      )}
                      dangerouslySetInnerHTML={{ 
                        __html: formatDescriptionContent(product.description) 
                      }}
                    />
                  </Card>
                </div>
              )}

              <Separator />

              {/* Product Attributes */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Product Attributes</h3>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  {product.article_type && (
                    <div>
                      <span className="text-muted-foreground">Article Type:</span>
                      <p className="font-medium">{product.article_type}</p>
                    </div>
                  )}
                  
                  {product.master_category && (
                    <div>
                      <span className="text-muted-foreground">Category:</span>
                      <p className="font-medium">{product.master_category}</p>
                    </div>
                  )}
                  
                  {product.sub_category && (
                    <div>
                      <span className="text-muted-foreground">Sub Category:</span>
                      <p className="font-medium">{product.sub_category}</p>
                    </div>
                  )}
                  
                  {product.gender && (
                    <div>
                      <span className="text-muted-foreground">Gender:</span>
                      <p className="font-medium">{product.gender}</p>
                    </div>
                  )}
                  
                  {product.base_colour && (
                    <div>
                      <span className="text-muted-foreground">Color:</span>
                      <p className="font-medium flex items-center gap-1">
                        <Palette className="h-4 w-4" />
                        {product.base_colour}
                      </p>
                    </div>
                  )}
                  
                  {product.season && (
                    <div>
                      <span className="text-muted-foreground">Season:</span>
                      <p className="font-medium flex items-center gap-1">
                        <Calendar className="h-4 w-4" />
                        {product.season}
                      </p>
                    </div>
                  )}
                  
                  {product.year && (
                    <div>
                      <span className="text-muted-foreground">Year:</span>
                      <p className="font-medium">{product.year}</p>
                    </div>
                  )}
                  
                  {product.usage && (
                    <div>
                      <span className="text-muted-foreground">Usage:</span>
                      <p className="font-medium flex items-center gap-1">
                        <TrendingUp className="h-4 w-4" />
                        {product.usage}
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <Separator />

              {/* Technical Information - Removed Product ID */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Technical Details</h3>
                
                <div className="bg-muted/50 rounded-lg p-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Similarity Score:</span>
                    <span className="font-medium">{(product.similarity_score * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>

              {/* Learning Section */}
              <Card className="p-4 bg-blue-50 border-blue-200">
                <h4 className="font-semibold mb-2 text-blue-900">🤖 How AI Found This Match</h4>
                <p className="text-sm text-blue-800">
                  Our AI model analyzes visual features like colors, patterns, shapes, and textures 
                  to calculate similarity scores. Higher scores indicate stronger visual resemblance 
                  to your search criteria using advanced computer vision technology.
                </p>
              </Card>
            </div>
          </div>
        </div>

        {/* Dialog Close Button */}
        <DialogClose asChild>
          <Button variant="outline" className="mt-6">
            Close Details
          </Button>
        </DialogClose>
      </DialogContent>
    </Dialog>
  )
} 