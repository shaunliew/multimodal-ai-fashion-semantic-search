/**
 * @fileoverview Download Utilities - Helper functions for downloading images and files
 * @learning Demonstrates proper file download handling with CORS support and comprehensive error handling
 * @concepts Blob handling, cross-origin downloads, file naming, error handling, fallback strategies
 */

/**
 * Learning: Enhanced image download function with comprehensive CORS handling
 * This approach tries multiple strategies to handle cross-origin images:
 * 1. Blob download with CORS headers
 * 2. Simple anchor download fallback
 * 3. Window.open fallback for stubborn CORS issues
 */
export const downloadImage = async (
  imageUrl: string, 
  filename: string,
  onProgress?: (progress: number) => void,
  onError?: (error: string) => void
): Promise<void> => {
  try {
    /**
     * Learning: Show initial progress
     * Let user know the download process has started
     */
    onProgress?.(5)

    /**
     * Learning: First attempt - Blob download with proper CORS handling
     * This works for images that have CORS headers configured
     */
    try {
      const response = await fetch(imageUrl, {
        method: 'GET',
        headers: {
          'Accept': 'image/*',
        },
        mode: 'cors', // Explicitly request CORS
        cache: 'no-cache', // Avoid cached responses that might not have CORS headers
      })

      onProgress?.(20)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      /**
       * Learning: Check if we can track download progress
       * Some servers provide Content-Length header for progress tracking
       */
      const contentLength = response.headers.get('content-length')
      const total = contentLength ? parseInt(contentLength, 10) : 0

      // Read the response as a stream to track progress
      const reader = response.body?.getReader()
      const chunks: Uint8Array[] = []
      let receivedLength = 0

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          
          if (done) break
          
          chunks.push(value)
          receivedLength += value.length
          
          // Update progress if we know the total size
          if (total > 0) {
            const progress = Math.round((receivedLength / total) * 80) + 20 // 20-100%
            onProgress?.(progress)
          } else {
            // Incremental progress without knowing total size
            const estimatedProgress = Math.min(20 + (receivedLength / 1024 / 1024) * 10, 90)
            onProgress?.(estimatedProgress)
          }
        }
      }

      // Combine all chunks into a single Uint8Array
      const allChunks = new Uint8Array(receivedLength)
      let position = 0
      for (const chunk of chunks) {
        allChunks.set(chunk, position)
        position += chunk.length
      }

      /**
       * Learning: Create blob with proper MIME type detection
       * Try to determine MIME type from response headers or URL
       */
      const contentType = response.headers.get('content-type') || getMimeTypeFromUrl(imageUrl)
      const blob = new Blob([allChunks], { type: contentType })
      
      // Create a temporary URL for the blob
      const blobUrl = URL.createObjectURL(blob)
      
      /**
       * Learning: Create and trigger download programmatically
       * This method works reliably across all modern browsers
       */
      const link = document.createElement('a')
      link.href = blobUrl
      link.download = filename
      
      // Temporarily add to DOM to ensure it works in all browsers
      document.body.appendChild(link)
      link.click()
      
      // Clean up: remove the link and revoke the blob URL
      document.body.removeChild(link)
      URL.revokeObjectURL(blobUrl)
      
      onProgress?.(100)
      
      console.log('Image downloaded successfully via blob:', filename)
      return

    } catch (blobError) {
      console.warn('Blob download failed, trying fallback methods:', blobError)
      
      /**
       * Learning: Check if this is a CORS error specifically
       * Different browsers report CORS errors differently
       */
      const errorMessage = blobError instanceof Error ? blobError.message : String(blobError)
      const isCorsError = errorMessage.includes('CORS') || 
                         errorMessage.includes('Failed to fetch') || 
                         errorMessage.includes('blocked by CORS') ||
                         errorMessage.includes('Cross-Origin Request Blocked')

      if (isCorsError) {
        console.log('CORS error detected, attempting alternative download methods...')
        
        /**
         * Learning: Fallback 1 - Simple anchor download
         * This works for some images even when fetch fails due to CORS
         */
        try {
          onProgress?.(80)
          
          const link = document.createElement('a')
          link.href = imageUrl
          link.download = filename
          link.target = '_blank' // Open in new tab as fallback
          link.rel = 'noopener noreferrer'
          
          // Some browsers require the link to be in DOM
          document.body.appendChild(link)
          link.click()
          document.body.removeChild(link)
          
          onProgress?.(100)
          console.log('Fallback anchor download initiated:', filename)
          return
          
        } catch (anchorError) {
          console.warn('Anchor download failed, trying window.open:', anchorError)
          
          /**
           * Learning: Fallback 2 - Window.open method
           * Last resort for CORS-protected images
           */
          try {
            onProgress?.(90)
            
            // Open image in new window/tab - user can manually save
            const newWindow = window.open(imageUrl, '_blank', 'noopener,noreferrer')
            
            if (newWindow) {
              onProgress?.(100)
              console.log('Opened image in new window for manual download:', filename)
              
              // Show user guidance
              setTimeout(() => {
                if (onError) {
                  onError('Image opened in new tab - please right-click and "Save As" to download')
                }
              }, 1000)
              return
            } else {
              throw new Error('Popup blocked or failed to open')
            }
            
          } catch (windowError) {
            console.error('All download methods failed:', windowError)
            throw new Error(`CORS policy prevents direct download. Original error: ${errorMessage}`)
          }
        }
      } else {
        // Re-throw non-CORS errors
        throw blobError
      }
    }
    
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown download error'
    console.error('Download failed:', errorMessage)
    
    /**
     * Learning: Provide helpful error message based on error type
     * Help users understand why the download failed and what they can do
     */
    let userMessage = errorMessage
    
    if (errorMessage.includes('CORS') || errorMessage.includes('Failed to fetch')) {
      userMessage = 'This image is protected by CORS policy and cannot be downloaded directly. This is common with external fashion websites for copyright protection.'
    } else if (errorMessage.includes('404') || errorMessage.includes('Not Found')) {
      userMessage = 'Image not found - the file may have been moved or deleted.'
    } else if (errorMessage.includes('403') || errorMessage.includes('Forbidden')) {
      userMessage = 'Access denied - the server does not allow downloading this image.'
    } else if (errorMessage.includes('network') || errorMessage.includes('Network')) {
      userMessage = 'Network error - please check your internet connection and try again.'
    }
    
    onError?.(userMessage)
    throw new Error(userMessage)
  }
}

/**
 * Learning: Helper function to determine MIME type from URL extension
 * Fallback when Content-Type header is not available
 */
const getMimeTypeFromUrl = (url: string): string => {
  const extension = url.split('.').pop()?.toLowerCase()
  
  const mimeTypes: Record<string, string> = {
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'webp': 'image/webp',
    'svg': 'image/svg+xml',
    'bmp': 'image/bmp',
    'tiff': 'image/tiff',
    'tif': 'image/tiff'
  }
  
  return mimeTypes[extension || ''] || 'image/jpeg'
}

/**
 * Learning: Generate clean, filesystem-safe filenames
 * Removes special characters that could cause issues on different operating systems
 */
export const generateImageFilename = (
  itemId: string,
  title: string,
  extension: string = 'jpg'
): string => {
  // Clean the title to be filesystem-safe
  const cleanTitle = title
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '') // Remove special characters
    .replace(/\s+/g, '-') // Replace spaces with hyphens
    .replace(/-+/g, '-') // Replace multiple hyphens with single
    .replace(/^-|-$/g, '') // Remove leading/trailing hyphens
    .substring(0, 50) // Limit length to avoid filesystem issues
  
  // Create timestamp for uniqueness
  const timestamp = new Date().toISOString().slice(0, 10) // YYYY-MM-DD format
  
  return `fashion-${itemId}-${cleanTitle}-${timestamp}.${extension}`
}

/**
 * Learning: Extract file extension from URL or MIME type
 * Helps determine the correct file extension for downloaded images
 */
export const getImageExtension = (url: string, mimeType?: string): string => {
  // Try to get extension from MIME type first
  if (mimeType) {
    const mimeToExt: Record<string, string> = {
      'image/jpeg': 'jpg',
      'image/jpg': 'jpg',
      'image/png': 'png',
      'image/gif': 'gif',
      'image/webp': 'webp',
      'image/svg+xml': 'svg',
      'image/bmp': 'bmp',
      'image/tiff': 'tiff'
    }
    
    if (mimeToExt[mimeType]) {
      return mimeToExt[mimeType]
    }
  }
  
  // Fallback to URL extension
  const urlExtension = url.split('.').pop()?.toLowerCase()
  const validExtensions = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'bmp', 'tiff', 'tif']
  
  if (urlExtension && validExtensions.includes(urlExtension)) {
    return urlExtension === 'jpeg' ? 'jpg' : urlExtension
  }
  
  // Default fallback
  return 'jpg'
}

/**
 * Learning: Utility function to check if URL might have CORS issues
 * Helps predict potential download problems
 */
export const isPotentialCorsIssue = (imageUrl: string): boolean => {
  try {
    const url = new URL(imageUrl)
    const currentDomain = window.location.hostname
    
    // Same domain is usually safe
    if (url.hostname === currentDomain) {
      return false
    }
    
    // Common domains that usually have CORS issues
    const problematicDomains = [
      'amazon.com',
      'ebay.com',
      'shopify.com',
      'woocommerce.com',
      'bigcommerce.com',
      'squarespace.com',
      'wordpress.com'
    ]
    
    return problematicDomains.some(domain => url.hostname.includes(domain))
    
  } catch {
    // If URL parsing fails, assume potential issues
    return true
  }
} 