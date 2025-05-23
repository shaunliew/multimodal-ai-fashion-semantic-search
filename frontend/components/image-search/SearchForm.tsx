/**
 * @fileoverview Search Form Component - Dual-mode search interface for semantic image search
 * @learning Demonstrates form handling, file uploads, and search mode switching in React
 * @concepts Text-to-image search, image-to-image search, file validation, accessibility
 */

'use client'

import React, { useCallback, useState } from 'react'
import { Search, Upload, X, Image as ImageIcon, Type } from 'lucide-react'
import Image from 'next/image'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { useImageSearchStore, SEARCH_MODES } from '@/stores/imageSearchStore'
import { cn } from '@/lib/utils'

interface SearchFormProps {
  className?: string
}

export const SearchForm = ({ className }: SearchFormProps) => {
  /**
   * Learning: Zustand store provides centralized state management
   * We destructure only the actions and state we need for this component
   */
  const {
    textQuery,
    uploadedImage,
    searchMode,
    isSearching,
    error,
    setTextQuery,
    setUploadedImage,
    setSearchMode,
    performTextSearch,
    performImageSearch,
    clearError
  } = useImageSearchStore()

  // Local state for file drag and drop
  const [isDragOver, setIsDragOver] = useState(false)

  /**
   * Learning: useCallback prevents unnecessary re-renders
   * Particularly important for file handling which can be expensive
   */
  const handleFileUpload = useCallback((file: File) => {
    /**
     * Learning: File validation is crucial for security and UX
     * We check file type and size before processing
     */
    if (!file.type.startsWith('image/')) {
      // Note: In production, you'd use a proper toast notification
      alert('Please upload an image file (JPG, PNG, WebP)')
      return
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      alert('Image file must be smaller than 10MB')
      return
    }

    setUploadedImage(file)
    clearError()
  }, [setUploadedImage, clearError])

  /**
   * Learning: Drag and drop provides modern UX for file uploads
   * These handlers manage the visual feedback during drag operations
   */
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }, [handleFileUpload])

  /**
   * Learning: Form submission with validation and error handling
   * Different search modes require different validation logic
   */
  const handleTextSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!textQuery.trim()) {
      return // Input validation handled by store
    }

    await performTextSearch(textQuery)
  }

  const handleImageSearch = async () => {
    if (!uploadedImage) {
      return // Validation handled by store
    }

    await performImageSearch(uploadedImage)
  }

  /**
   * Learning: Tab switching with mode management
   * Updates global state and clears irrelevant data
   */
  const handleTabChange = (value: string) => {
    if (value === 'text') {
      setSearchMode(SEARCH_MODES.TEXT_TO_IMAGE)
    } else if (value === 'image') {
      setSearchMode(SEARCH_MODES.IMAGE_TO_IMAGE)
    }
  }

  return (
    <Card className={cn("w-full max-w-2xl mx-auto", className)}>
      <CardContent className="p-6">
        {/* Error Display */}
        {error && (
          <Alert className="mb-4" variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Search Mode Tabs */}
        <Tabs 
          value={searchMode === SEARCH_MODES.TEXT_TO_IMAGE ? 'text' : 'image'} 
          onValueChange={handleTabChange}
          className="w-full"
        >
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="text" className="flex items-center gap-2">
              <Type className="h-4 w-4" />
              Text Search
            </TabsTrigger>
            <TabsTrigger value="image" className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4" />
              Image Search
            </TabsTrigger>
          </TabsList>

          {/* Text-to-Image Search */}
          <TabsContent value="text" className="space-y-4">
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Search by Description</h3>
              <p className="text-sm text-muted-foreground">
                Describe the fashion item you&apos;re looking for using natural language
              </p>
            </div>

            <form onSubmit={handleTextSearch} className="space-y-4">
              <div className="relative">
                <Input
                  type="text"
                  placeholder="e.g., &quot;red summer dress&quot;, &quot;vintage leather jacket&quot;, &quot;blue denim jeans&quot;..."
                  value={textQuery}
                  onChange={(e) => setTextQuery(e.target.value)}
                  disabled={isSearching}
                  className="pr-12 h-12 text-base"
                  aria-label="Fashion search query"
                  aria-describedby="text-search-help"
                />
                <Button
                  type="submit"
                  size="sm"
                  disabled={!textQuery.trim() || isSearching}
                  className="absolute right-1 top-1 h-10"
                  aria-label={isSearching ? 'Searching...' : 'Search fashion items'}
                >
                  <Search className="h-4 w-4" />
                </Button>
              </div>
              
              <p id="text-search-help" className="text-xs text-muted-foreground">
                💡 Use descriptive terms like colors, materials, styles, or occasions for better results
              </p>
            </form>
          </TabsContent>

          {/* Image-to-Image Search */}
          <TabsContent value="image" className="space-y-4">
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Search by Image</h3>
              <p className="text-sm text-muted-foreground">
                Upload a fashion image to find visually similar items
              </p>
            </div>

            {/* File Upload Area */}
            <div
              className={cn(
                "relative border-2 border-dashed rounded-lg p-8 transition-colors",
                isDragOver ? "border-primary bg-primary/5" : "border-muted-foreground/25",
                uploadedImage ? "border-green-500 bg-green-50" : ""
              )}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {uploadedImage ? (
                // Image Preview
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="relative w-16 h-16 rounded-lg overflow-hidden bg-muted">
                      <Image
                        src={URL.createObjectURL(uploadedImage)}
                        alt="Uploaded fashion image"
                        width={64}
                        height={64}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div>
                      <p className="font-medium">{uploadedImage.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(uploadedImage.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      onClick={handleImageSearch}
                      disabled={isSearching}
                      className="flex items-center gap-2"
                    >
                      <Search className="h-4 w-4" />
                      Find Similar
                    </Button>
                    <Button
                      onClick={() => setUploadedImage(null)}
                      variant="outline"
                      size="icon"
                      aria-label="Remove uploaded image"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ) : (
                // Upload Prompt
                <div className="text-center space-y-4">
                  <Upload className="h-12 w-12 mx-auto text-muted-foreground" />
                  <div className="space-y-2">
                    <p className="text-base font-medium">
                      Drop your fashion image here
                    </p>
                    <p className="text-sm text-muted-foreground">
                      or click to browse files
                    </p>
                  </div>
                  
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      const input = document.createElement('input')
                      input.type = 'file'
                      input.accept = 'image/*'
                      input.onchange = (e) => {
                        const file = (e.target as HTMLInputElement).files?.[0]
                        if (file) handleFileUpload(file)
                      }
                      input.click()
                    }}
                    className="mt-2"
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Choose Image
                  </Button>
                </div>
              )}
            </div>

            <p className="text-xs text-muted-foreground">
              📸 Supported formats: JPG, PNG, WebP (max 10MB) • Best results with clear fashion item photos
            </p>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
} 