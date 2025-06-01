/**
 * @fileoverview Main Search Interface - Combines all search components into unified experience
 * @learning Demonstrates component composition, layout management, and responsive design
 * @concepts Layout composition, state coordination, responsive grid systems, mobile drawer UX
 */

'use client'

import React, { useState } from 'react'
import { Settings2, Grid3X3, List } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { SearchForm } from './SearchForm'
import { SearchResults } from './SearchResults'
import { SearchFilters } from './SearchFilters'
import { useImageSearchStore } from '@/stores/imageSearchStore'
import { cn } from '@/lib/utils'

export const SearchInterface = () => {
  /**
   * Learning: Local UI state vs Global state management
   * - showFilters: Local UI state (only affects this component)
   * - search data: Global state (shared across components via Zustand)
   * - Mobile: Use drawer state instead of inline filters
   */
  const [showFilters, setShowFilters] = useState(false)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false)
  
  const { results, isSearching, filters } = useImageSearchStore()

  /**
   * Learning: Count active filters for mobile badge display
   * Helps users understand when filters are applied on mobile
   */
  const getActiveFilterCount = () => {
    let count = 0
    if (filters.minSimilarity !== 0.7) count++ // Default is 0.7
    if (filters.maxResults !== 20) count++ // Default is 20
    if (filters.category) count++
    if (filters.tags && filters.tags.length > 0) count++
    return count
  }

  /**
   * Learning: Responsive layout with CSS Grid
   * We use different layouts for mobile vs desktop
   * Mobile: Drawer for filters, Desktop: Sidebar
   */
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Search Form Section */}
      <div className="mb-8">
        <SearchForm />
      </div>

      {/* Results Section */}
      {(results.length > 0 || isSearching) && (
        <div className="space-y-6">
          {/* Results Controls */}
          <div className="flex flex-col sm:flex-row gap-4 sm:items-center sm:justify-between">
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-semibold">Search Results</h2>
              {results.length > 0 && (
                <span className="text-sm text-muted-foreground">
                  ({results.length} items found)
                </span>
              )}
            </div>
            
            <div className="flex items-center gap-2">
              {/* View Mode Toggle */}
              <div className="flex items-center border rounded-lg p-1">
                <Button
                  variant={viewMode === 'grid' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('grid')}
                  className="h-8 px-3"
                >
                  <Grid3X3 className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('list')}
                  className="h-8 px-3"
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>

              {/* Mobile Filters Drawer */}
              <div className="lg:hidden">
                <Sheet open={mobileFiltersOpen} onOpenChange={setMobileFiltersOpen}>
                  <SheetTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2"
                    >
                      <Settings2 className="h-4 w-4" />
                      Filters
                      {getActiveFilterCount() > 0 && (
                        <span className="bg-primary text-primary-foreground text-xs rounded-full h-5 w-5 flex items-center justify-center">
                          {getActiveFilterCount()}
                        </span>
                      )}
                    </Button>
                  </SheetTrigger>
                  <SheetContent side="right" className="w-full sm:w-96 overflow-y-auto">
                    <SheetHeader>
                      <SheetTitle className="flex items-center gap-2">
                        <Settings2 className="h-5 w-5" />
                        Search Filters
                      </SheetTitle>
                    </SheetHeader>
                    <div className="mt-6">
                      <SearchFilters />
                    </div>
                  </SheetContent>
                </Sheet>
              </div>

              {/* Desktop Filters Toggle */}
              <div className="hidden lg:block">
                <Button
                  variant={showFilters ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setShowFilters(!showFilters)}
                  className="flex items-center gap-2"
                >
                  <Settings2 className="h-4 w-4" />
                  Filters
                  {getActiveFilterCount() > 0 && (
                    <span className="bg-white text-primary text-xs rounded-full h-5 w-5 flex items-center justify-center ml-1">
                      {getActiveFilterCount()}
                    </span>
                  )}
                </Button>
              </div>
            </div>
          </div>

          {/* Main Content Layout */}
          <div className={cn(
            "grid gap-6",
            // Learning: Conditional grid layouts based on filter visibility
            // Mobile: Always full width (drawer handles filters)
            // Desktop: Sidebar when filters are shown
            showFilters 
              ? "lg:grid-cols-[300px_1fr]" // Sidebar + main content on desktop
              : "grid-cols-1"              // Full width
          )}>
            {/* Desktop Filters Sidebar */}
            {showFilters && (
              <div className={cn(
                "hidden lg:block space-y-6",
                "order-2 lg:order-1", // Ensure proper order
              )}>
                <SearchFilters />
              </div>
            )}

            {/* Results Area */}
            <div className={cn(
              "space-y-6",
              showFilters ? "order-1 lg:order-2" : ""
            )}>
              <SearchResults 
                viewMode={viewMode}
                className={cn(
                  // Learning: Dynamic grid based on view mode
                  viewMode === 'list' && "max-w-none"
                )} 
              />
            </div>
          </div>
        </div>
      )}

      {/* Getting Started Guide - Show when no search performed */}
      {results.length === 0 && !isSearching && (
        <div className="mt-16">
          <GettingStartedGuide />
        </div>
      )}
    </div>
  )
}

/**
 * Getting Started Guide Component
 * Helps users understand how to use the semantic search features
 */
const GettingStartedGuide = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          How Semantic Fashion Search Works
        </h2>
        <p className="text-lg text-gray-600">
          Experience the power of AI-driven fashion discovery
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
        {/* Text Search Guide */}
        <div className="bg-white rounded-xl p-6 shadow-sm border">
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">💬</span>
          </div>
          <h3 className="text-xl font-semibold mb-3">Text-to-Image Search</h3>
          <p className="text-gray-600 mb-4">
            Describe fashion items using natural language and find visually matching products.
          </p>
          <div className="space-y-2">
            <div className="text-sm">
              <span className="font-medium">Try:</span>
              <ul className="list-disc list-inside text-gray-600 mt-1 space-y-1">
                <li>red flowy summer dress</li>
                <li>vintage leather jacket brown</li>
                <li>minimalist white sneakers</li>
                <li>striped oversized sweater</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Image Search Guide */}
        <div className="bg-white rounded-xl p-6 shadow-sm border">
          <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">📸</span>
          </div>
          <h3 className="text-xl font-semibold mb-3">Image-to-Image Search</h3>
          <p className="text-gray-600 mb-4">
            Upload a fashion image to discover visually similar items using computer vision.
          </p>
          <div className="space-y-2">
            <div className="text-sm">
              <span className="font-medium">Best results with:</span>
              <ul className="list-disc list-inside text-gray-600 mt-1 space-y-1">
                <li>Clear, well-lit product photos</li>
                <li>Single fashion item focus</li>
                <li>Minimal background distractions</li>
                <li>High-resolution images</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Overview */}
      <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl p-8">
        <h3 className="text-2xl font-bold text-center mb-6">
          Under the Hood: AI Technology
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center space-y-3">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
              <span className="text-2xl">🧠</span>
            </div>
            <h4 className="font-semibold">CLIP Embeddings</h4>
            <p className="text-sm text-gray-600">
              Neural networks convert text and images into semantic vector representations
            </p>
          </div>
          
          <div className="text-center space-y-3">
            <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto">
              <span className="text-2xl">📊</span>
            </div>
            <h4 className="font-semibold">Vector Database</h4>
            <p className="text-sm text-gray-600">
              MongoDB Atlas Vector Search enables fast similarity comparisons at scale
            </p>
          </div>
          
          <div className="text-center space-y-3">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
              <span className="text-2xl">🎯</span>
            </div>
            <h4 className="font-semibold">Cosine Similarity</h4>
            <p className="text-sm text-gray-600">
              Mathematical matching based on semantic meaning rather than keyword matching
            </p>
          </div>
        </div>
      </div>
    </div>
  )
} 