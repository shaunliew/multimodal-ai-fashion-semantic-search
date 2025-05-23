/**
 * @fileoverview Search Filters Component - Advanced filtering for semantic search results
 * @learning Demonstrates how filters work with vector search and similarity thresholds
 * @concepts Vector similarity thresholds, result filtering, search optimization
 */

'use client'

import React from 'react'
import { Filter, RotateCcw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { useImageSearchStore } from '@/stores/imageSearchStore'
import { cn } from '@/lib/utils'

interface SearchFiltersProps {
  className?: string
}

export const SearchFilters = ({ className }: SearchFiltersProps) => {
  const { filters, setFilters, results } = useImageSearchStore()

  /**
   * Learning: Similarity threshold is crucial for vector search quality
   * - Higher threshold (0.8+): Only very similar results, fewer items
   * - Lower threshold (0.6+): More diverse results, might include less relevant items
   * - Sweet spot is usually 0.7 for fashion items
   */
  const handleSimilarityChange = (threshold: number) => {
    setFilters({ minSimilarity: threshold })
  }

  /**
   * Learning: Result count affects both performance and user experience
   * Vector search is computationally expensive, so limiting results helps
   */
  const handleMaxResultsChange = (maxResults: number) => {
    setFilters({ maxResults })
  }

  /**
   * Learning: Reset filters to default values for fresh search experience
   */
  const handleResetFilters = () => {
    setFilters({
      minSimilarity: 0.7,
      maxResults: 20,
      category: undefined,
      tags: undefined
    })
  }

  /**
   * Pre-defined categories for fashion items
   * In production, these would come from your product database
   */
  const categories = [
    'All Categories',
    'Dresses',
    'Tops & Blouses', 
    'Pants & Jeans',
    'Skirts',
    'Jackets & Coats',
    'Shoes',
    'Accessories',
    'Bags'
  ]

  const similarityLevels = [
    { value: 0.9, label: 'Excellent (90%+)', description: 'Nearly identical items' },
    { value: 0.8, label: 'Very Similar (80%+)', description: 'Very close matches' },
    { value: 0.7, label: 'Similar (70%+)', description: 'Good matches (recommended)' },
    { value: 0.6, label: 'Somewhat Similar (60%+)', description: 'Broader results' },
    { value: 0.5, label: 'Loosely Similar (50%+)', description: 'Very diverse results' }
  ]

  const resultCountOptions = [10, 20, 50, 100]

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Filter className="h-5 w-5" />
          Search Filters
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Refine your semantic search results
        </p>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Similarity Threshold */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium">Similarity Threshold</h4>
            <Badge variant="outline">
              {Math.round(filters.minSimilarity * 100)}%
            </Badge>
          </div>
          
          <p className="text-xs text-muted-foreground">
            Higher values show only very similar items, lower values show more diverse results
          </p>
          
          <div className="space-y-2">
            {similarityLevels.map((level) => (
              <div
                key={level.value}
                className={cn(
                  "flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors",
                  filters.minSimilarity === level.value
                    ? "border-primary bg-primary/5"
                    : "border-border hover:bg-muted/50"
                )}
                onClick={() => handleSimilarityChange(level.value)}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <div className={cn(
                      "w-3 h-3 rounded-full border-2",
                      filters.minSimilarity === level.value
                        ? "border-primary bg-primary"
                        : "border-muted-foreground"
                    )} />
                    <span className="font-medium text-sm">{level.label}</span>
                  </div>
                  <p className="text-xs text-muted-foreground ml-5">
                    {level.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <Separator />

        {/* Result Count */}
        <div className="space-y-3">
          <h4 className="font-medium">Maximum Results</h4>
          <p className="text-xs text-muted-foreground">
            Limit the number of results to improve loading performance
          </p>
          
          <div className="flex flex-wrap gap-2">
            {resultCountOptions.map((count) => (
              <Button
                key={count}
                variant={filters.maxResults === count ? "default" : "outline"}
                size="sm"
                onClick={() => handleMaxResultsChange(count)}
                className="text-xs"
              >
                {count} items
              </Button>
            ))}
          </div>
        </div>

        <Separator />

        {/* Category Filter */}
        <div className="space-y-3">
          <h4 className="font-medium">Category</h4>
          <p className="text-xs text-muted-foreground">
            Filter by fashion category for more focused results
          </p>
          
          <div className="grid grid-cols-2 gap-2">
            {categories.map((category) => {
              const isSelected = category === 'All Categories' 
                ? !filters.category 
                : filters.category === category.toLowerCase().replace(/\s+/g, '_')
              
              return (
                <Button
                  key={category}
                  variant={isSelected ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    const newCategory = category === 'All Categories' 
                      ? undefined 
                      : category.toLowerCase().replace(/\s+/g, '_')
                    setFilters({ category: newCategory })
                  }}
                  className="text-xs justify-start"
                >
                  {category}
                </Button>
              )
            })}
          </div>
        </div>

        <Separator />

        {/* Filter Summary & Reset */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium">Active Filters</h4>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleResetFilters}
              className="text-xs"
            >
              <RotateCcw className="h-3 w-3 mr-1" />
              Reset
            </Button>
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Similarity:</span>
              <span>{Math.round(filters.minSimilarity * 100)}%+</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Max Results:</span>
              <span>{filters.maxResults}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Category:</span>
              <span>{filters.category || 'All'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Current Results:</span>
              <span className="font-medium">{results.length}</span>
            </div>
          </div>
        </div>

        {/* Performance Tips */}
        <div className="bg-muted/50 rounded-lg p-3">
          <h5 className="font-medium text-sm mb-2">💡 Search Tips</h5>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Higher similarity = fewer, more accurate results</li>
            <li>• Lower similarity = more diverse, exploratory results</li>
            <li>• Fewer max results = faster loading times</li>
            <li>• Category filters help narrow focus</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
} 