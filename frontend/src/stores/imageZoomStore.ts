/**
 * @fileoverview Image Zoom Store - Zustand state management for image zoom modal functionality
 * @learning Demonstrates complex UI state management with Zustand for image interactions
 * @concepts Zoom controls, pan interactions, rotation state, modal patterns, performance optimization
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

/**
 * Learning: Position interface for zoom and pan coordinates
 * Using explicit types helps with debugging and prevents coordinate system errors
 */
interface Position {
  x: number
  y: number
}

/**
 * Learning: Comprehensive zoom modal state interface
 * Centralizing all zoom-related state improves debugging and enables cross-component sharing
 */
interface ImageZoomState {
  // Zoom and transformation state
  zoom: number
  rotation: number
  position: Position
  
  // Interaction state
  isDragging: boolean
  dragStart: Position
  
  // Modal state
  isOpen: boolean
  currentImageUrl: string
  currentImageAlt: string
  currentItemId: string
  currentItemTitle: string
  
  // Zoom control actions
  setZoom: (zoom: number) => void
  zoomIn: () => void
  zoomOut: () => void
  
  // Rotation actions
  rotate: () => void
  setRotation: (rotation: number) => void
  
  // Position and panning actions
  setPosition: (position: Position) => void
  startDrag: (startPosition: Position) => void
  updateDragPosition: (currentPosition: Position) => void
  stopDrag: () => void
  
  // Modal actions
  openZoomModal: (imageUrl: string, imageAlt: string, itemId: string, itemTitle?: string) => void
  closeZoomModal: () => void
  
  // Reset and utility actions
  resetView: () => void
  resetAll: () => void
  
  // Wheel zoom action
  handleWheelZoom: (delta: number) => void
}

/**
 * Learning: Zustand store for centralized zoom modal state management
 * Benefits over useState:
 * - Centralized state for better debugging
 * - Actions encapsulate business logic
 * - Easy to share state across components
 * - DevTools integration for better debugging
 * - No prop drilling needed
 */
export const useImageZoomStore = create<ImageZoomState>()(
  devtools(
    (set, get) => ({
      // Initial state
      zoom: 1,
      rotation: 0,
      position: { x: 0, y: 0 },
      isDragging: false,
      dragStart: { x: 0, y: 0 },
      isOpen: false,
      currentImageUrl: '',
      currentImageAlt: '',
      currentItemId: '',
      currentItemTitle: '',

      // Zoom control actions
      setZoom: (zoom: number) => {
        /**
         * Learning: Bounded zoom with automatic position reset
         * Prevents extreme zoom levels and resets position when zooming out to fit
         */
        const boundedZoom = Math.min(Math.max(zoom, 0.5), 5)
        
        set((state) => ({
          zoom: boundedZoom,
          // Reset position when zooming out to fit
          position: boundedZoom <= 1 ? { x: 0, y: 0 } : state.position
        }))
      },

      zoomIn: () => {
        const currentZoom = get().zoom
        get().setZoom(currentZoom * 1.5)
      },

      zoomOut: () => {
        const currentZoom = get().zoom
        get().setZoom(currentZoom / 1.5)
      },

      // Rotation actions
      rotate: () => {
        set((state) => ({
          rotation: (state.rotation + 90) % 360
        }))
      },

      setRotation: (rotation: number) => {
        set({ rotation: rotation % 360 })
      },

      // Position and panning actions
      setPosition: (position: Position) => {
        set({ position })
      },

      startDrag: (startPosition: Position) => {
        /**
         * Learning: Only allow dragging when zoomed in
         * This prevents unnecessary pan operations when the image fits the view
         */
        const { zoom, position } = get()
        if (zoom > 1) {
          set({
            isDragging: true,
            dragStart: {
              x: startPosition.x - position.x,
              y: startPosition.y - position.y
            }
          })
        }
      },

      updateDragPosition: (currentPosition: Position) => {
        const { isDragging, dragStart, zoom } = get()
        if (isDragging && zoom > 1) {
          set({
            position: {
              x: currentPosition.x - dragStart.x,
              y: currentPosition.y - dragStart.y
            }
          })
        }
      },

      stopDrag: () => {
        set({ isDragging: false })
      },

      // Modal actions
      openZoomModal: (imageUrl: string, imageAlt: string, itemId: string, itemTitle = '') => {
        /**
         * Learning: Reset zoom state when opening modal
         * Ensures consistent experience for each image view
         */
        set({
          isOpen: true,
          currentImageUrl: imageUrl,
          currentImageAlt: imageAlt,
          currentItemId: itemId,
          currentItemTitle: itemTitle,
          // Reset view state for new image
          zoom: 1,
          rotation: 0,
          position: { x: 0, y: 0 },
          isDragging: false,
          dragStart: { x: 0, y: 0 }
        })
      },

      closeZoomModal: () => {
        set({
          isOpen: false,
          currentImageUrl: '',
          currentImageAlt: '',
          currentItemId: '',
          currentItemTitle: ''
        })
      },

      // Reset and utility actions
      resetView: () => {
        /**
         * Learning: Reset only view-related state, keep modal open
         * Useful for "reset view" button functionality
         */
        set({
          zoom: 1,
          rotation: 0,
          position: { x: 0, y: 0 },
          isDragging: false,
          dragStart: { x: 0, y: 0 }
        })
      },

      resetAll: () => {
        /**
         * Learning: Complete reset including modal state
         * Useful for cleanup or emergency reset
         */
        set({
          zoom: 1,
          rotation: 0,
          position: { x: 0, y: 0 },
          isDragging: false,
          dragStart: { x: 0, y: 0 },
          isOpen: false,
          currentImageUrl: '',
          currentImageAlt: '',
          currentItemId: '',
          currentItemTitle: ''
        })
      },

      // Wheel zoom action
      handleWheelZoom: (deltaY: number) => {
        /**
         * Learning: Smooth wheel zoom with proper scaling
         * Provides intuitive zoom experience similar to native image viewers
         */
        const delta = deltaY > 0 ? 0.9 : 1.1
        const currentZoom = get().zoom
        get().setZoom(currentZoom * delta)
      }
    }),
    { 
      name: 'image-zoom-store', // DevTools identifier
      /**
       * Learning: Store separation for specialized functionality
       * Image zoom state is separate from other UI state for better organization
       */
    }
  )
) 