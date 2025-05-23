/**
 * @fileoverview useMediaQuery hook for responsive design detection
 * @learning Demonstrates custom React hooks for media query-based responsive behavior
 * @concepts React hooks, window.matchMedia API, responsive design patterns
 */

'use client'

import { useState, useEffect } from 'react'

/**
 * Learning: Custom hook for responsive design without CSS-in-JS dependencies
 * This pattern allows components to change behavior based on screen size
 * 
 * @param query CSS media query string (e.g., '(min-width: 768px)')
 * @returns boolean indicating if the media query matches
 */
export const useMediaQuery = (query: string): boolean => {
  const [matches, setMatches] = useState(false)

  useEffect(() => {
    /**
     * Learning: window.matchMedia provides JavaScript access to CSS media queries
     * This enables responsive behavior beyond just CSS styling
     */
    const media = window.matchMedia(query)
    
    // Set initial value
    if (media.matches !== matches) {
      setMatches(media.matches)
    }
    
    // Create listener for changes
    const listener = (event: MediaQueryListEvent) => {
      setMatches(event.matches)
    }
    
    // Add listener - modern browsers support addEventListener
    if (media.addEventListener) {
      media.addEventListener('change', listener)
    } else {
      // Fallback for older browsers
      media.addListener(listener)
    }
    
    // Cleanup function
    return () => {
      if (media.removeEventListener) {
        media.removeEventListener('change', listener)
      } else {
        media.removeListener(listener)
      }
    }
  }, [matches, query])

  return matches
} 