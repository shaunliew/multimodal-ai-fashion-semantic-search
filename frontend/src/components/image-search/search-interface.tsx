/**
 * @fileoverview Main search interface component integrating with FastAPI backend
 * @learning Demonstrates React Query integration, real-time API calls, and responsive UI
 * @concepts API integration, state management, error handling, performance optimization
 */

'use client'

import { useState, useCallback, useEffect } from 'react'
import Image from 'next/image'
import { Upload, Search, Link2, Loader2, AlertCircle } from 'lucide-react'
import { useSemanticSearch } from '@/hooks/use-api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { SearchResults } from './search-results'
import { SearchStats } from './search-stats'
import type { SearchMode } from '@/types/api'

/**
 * Main search interface component
 * @learning Integrates with FastAPI backend using React Query hooks
 */
export function SearchInterface() {
  // Search state
  const [searchMode, setSearchMode] = useState<SearchMode>('text')
  const [textQuery, setTextQuery] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState('')
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [numberOfResults, setNumberOfResults] = useState(20)

  // Use the combined search hook
  const { search, isSearching, results, health, stats, isHealthy, error } = useSemanticSearch()

  /**
   * Handle text search submission
   * @learning Demonstrates API integration with error handling
   */
  const handleTextSearch = useCallback(async () => {
    if (!textQuery.trim()) return

    await search({
      mode: 'text',
      query: textQuery,
      numberOfResults,
    })
  }, [textQuery, numberOfResults, search])

  /**
   * Handle image file selection and preview
   * @learning File handling with preview generation
   */
  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setImageFile(file)
    
    // Generate preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setImagePreview(reader.result as string)
    }
    reader.readAsDataURL(file)
  }, [])

  /**
   * Handle image search submission
   * @learning Multipart form data submission for images
   */
  const handleImageSearch = useCallback(async () => {
    if (!imageFile) return

    await search({
      mode: 'image',
      imageFile,
      numberOfResults,
      excludeReference: true,
    })
  }, [imageFile, numberOfResults, search])

  /**
   * Handle URL-based image search
   * @learning Alternative image search method using URLs
   */
  const handleUrlSearch = useCallback(async () => {
    if (!imageUrl.trim()) return

    await search({
      mode: 'url',
      imageUrl,
      numberOfResults,
      excludeReference: true,
    })
  }, [imageUrl, numberOfResults, search])

  /**
   * Handle form submission based on active tab
   * @learning Unified submission handler for different search modes
   */
  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()

      switch (searchMode) {
        case 'text':
          handleTextSearch()
          break
        case 'image':
          handleImageSearch()
          break
        case 'url':
          handleUrlSearch()
          break
      }
    },
    [searchMode, handleTextSearch, handleImageSearch, handleUrlSearch]
  )

  // Show health status in development
  useEffect(() => {
    if (process.env.NODE_ENV === 'development' && health) {
      console.log('API Health Status:', health)
    }
  }, [health])

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Health Status Alert */}
      {!isHealthy && health && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-yellow-600" />
            <p className="text-sm text-yellow-800">
              Search service is experiencing issues. Some features may be unavailable.
            </p>
          </div>
        </div>
      )}

      {/* Search Form */}
      <Card className="p-6 mb-8">
        <form onSubmit={handleSubmit} className="space-y-6">
          <Tabs
            value={searchMode}
            onValueChange={(value) => setSearchMode(value as SearchMode)}
            className="w-full"
          >
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="text" className="flex items-center gap-2">
                <Search className="h-4 w-4" />
                Text Search
              </TabsTrigger>
              <TabsTrigger value="image" className="flex items-center gap-2">
                <Upload className="h-4 w-4" />
                Image Upload
              </TabsTrigger>
              <TabsTrigger value="url" className="flex items-center gap-2">
                <Link2 className="h-4 w-4" />
                Image URL
              </TabsTrigger>
            </TabsList>

            <TabsContent value="text" className="space-y-4">
              <div>
                <Label htmlFor="text-query">Describe what you&apos;re looking for</Label>
                <Textarea
                  id="text-query"
                  placeholder="e.g., red summer dress with floral pattern, casual denim jacket, elegant black evening gown..."
                  value={textQuery}
                  onChange={(e) => setTextQuery(e.target.value)}
                  className="mt-2 min-h-[100px]"
                  disabled={isSearching}
                />
                <p className="text-sm text-muted-foreground mt-2">
                  Use natural language to describe colors, patterns, styles, or specific items
                </p>
              </div>
            </TabsContent>

            <TabsContent value="image" className="space-y-4">
              <div>
                <Label htmlFor="image-upload">Upload a reference image</Label>
                <div className="mt-2 space-y-4">
                  <Input
                    id="image-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    disabled={isSearching}
                    className="cursor-pointer"
                  />
                  
                  {imagePreview && (
                    <div className="relative w-full max-w-sm mx-auto">
                      <Image
                        src={imagePreview}
                        alt="Preview"
                        width={400}
                        height={400}
                        className="w-full h-auto rounded-lg shadow-md object-contain"
                      />
                      <Button
                        type="button"
                        variant="destructive"
                        size="sm"
                        className="absolute top-2 right-2"
                        onClick={() => {
                          setImageFile(null)
                          setImagePreview(null)
                        }}
                      >
                        Remove
                      </Button>
                    </div>
                  )}
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  Upload an image to find visually similar fashion items
                </p>
              </div>
            </TabsContent>

            <TabsContent value="url" className="space-y-4">
              <div>
                <Label htmlFor="image-url">Image URL</Label>
                <Input
                  id="image-url"
                  type="url"
                  placeholder="https://example.com/fashion-image.jpg"
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  className="mt-2"
                  disabled={isSearching}
                />
                <p className="text-sm text-muted-foreground mt-2">
                  Paste a direct link to an image from the web
                </p>
              </div>
            </TabsContent>
          </Tabs>

          {/* Search Options */}
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <Label htmlFor="num-results">Number of results</Label>
              <Input
                id="num-results"
                type="number"
                min={1}
                max={100}
                value={numberOfResults}
                onChange={(e) => setNumberOfResults(parseInt(e.target.value) || 20)}
                className="mt-2 max-w-[120px]"
                disabled={isSearching}
              />
            </div>

            <Button
              type="submit"
              size="lg"
              disabled={
                isSearching ||
                !isHealthy ||
                (searchMode === 'text' && !textQuery.trim()) ||
                (searchMode === 'image' && !imageFile) ||
                (searchMode === 'url' && !imageUrl.trim())
              }
              className="min-w-[120px]"
            >
              {isSearching ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />
                  Search
                </>
              )}
            </Button>
          </div>
        </form>
      </Card>

      {/* Statistics Dashboard */}
      {stats && (
        <SearchStats stats={stats} className="mb-8" />
      )}

      {/* Error Display */}
      {error && (
        <Card className="p-6 mb-8 border-red-200 bg-red-50">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-red-900">Search Error</h3>
              <p className="text-sm text-red-700 mt-1">{error.message}</p>
            </div>
          </div>
        </Card>
      )}

      {/* Search Results */}
      {results && !error && (
        <SearchResults
          results={results}
          isLoading={isSearching}
        />
      )}
    </div>
  )
} 