/**
 * @fileoverview Liked Items Store - Zustand state management for user favorites/wishlist
 * @learning Demonstrates persistent state management for user preferences with localStorage
 * @concepts User preferences, localStorage persistence, set operations, optimistic updates
 */

import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'

/**
 * Learning: Liked items interface for type safety
 * We store minimal data to keep localStorage light and fast
 */
interface LikedItem {
  id: string
  imageUrl: string
  title: string
  likedAt: Date
}

interface LikedItemsState {
  // State
  likedItems: Set<string> // Using Set for O(1) lookup performance
  likedItemsData: LikedItem[] // Full data for displaying liked items list
  
  // Actions
  toggleLike: (itemId: string, itemData?: Partial<LikedItem>) => void
  isLiked: (itemId: string) => boolean
  getLikedItems: () => LikedItem[]
  clearAllLikes: () => void
  removeLike: (itemId: string) => void
}

/**
 * Learning: Zustand store with persistence for user preferences
 * localStorage ensures likes persist across browser sessions
 * DevTools help debug state changes during development
 */
export const useLikedItemsStore = create<LikedItemsState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        likedItems: new Set<string>(),
        likedItemsData: [],

        // Toggle like status for an item
        toggleLike: (itemId: string, itemData?: Partial<LikedItem>) => {
          const currentLiked = get().likedItems
          const currentData = get().likedItemsData
          
          /**
           * Learning: Optimistic updates for better UX
           * We immediately update the UI, assuming the operation will succeed
           */
          if (currentLiked.has(itemId)) {
            // Remove from liked items
            const newLiked = new Set(currentLiked)
            newLiked.delete(itemId)
            
            const newData = currentData.filter(item => item.id !== itemId)
            
            set({ 
              likedItems: newLiked, 
              likedItemsData: newData 
            })
          } else {
            // Add to liked items
            const newLiked = new Set(currentLiked)
            newLiked.add(itemId)
            
            /**
             * Learning: Only add to data array if we have the item details
             * This prevents incomplete data in the liked items list
             */
            let newData = currentData
            if (itemData) {
              const likedItem: LikedItem = {
                id: itemId,
                imageUrl: itemData.imageUrl || '',
                title: itemData.title || 'Untitled Item',
                likedAt: new Date()
              }
              newData = [...currentData, likedItem]
            }
            
            set({ 
              likedItems: newLiked, 
              likedItemsData: newData 
            })
          }
        },

        // Check if an item is liked (O(1) lookup using Set)
        isLiked: (itemId: string): boolean => {
          return get().likedItems.has(itemId)
        },

        // Get all liked items with full data
        getLikedItems: (): LikedItem[] => {
          return get().likedItemsData.sort((a, b) => 
            new Date(b.likedAt).getTime() - new Date(a.likedAt).getTime()
          )
        },

        // Remove a specific like
        removeLike: (itemId: string) => {
          const currentLiked = get().likedItems
          const currentData = get().likedItemsData
          
          if (currentLiked.has(itemId)) {
            const newLiked = new Set(currentLiked)
            newLiked.delete(itemId)
            
            const newData = currentData.filter(item => item.id !== itemId)
            
            set({ 
              likedItems: newLiked, 
              likedItemsData: newData 
            })
          }
        },

        // Clear all likes (useful for logout or reset functionality)
        clearAllLikes: () => {
          set({ 
            likedItems: new Set<string>(), 
            likedItemsData: [] 
          })
        }
      }),
      {
        name: 'liked-items-storage', // localStorage key
        /**
         * Learning: Custom storage transformations for Set serialization
         * Sets don't serialize to JSON natively, so we convert to/from arrays
         */
        partialize: (state) => ({
          likedItems: Array.from(state.likedItems),
          likedItemsData: state.likedItemsData
        }),
        onRehydrateStorage: () => (state) => {
          if (state) {
            // Convert array back to Set after rehydration from localStorage
            state.likedItems = new Set(state.likedItems as unknown as string[])
          }
        },
        version: 1, // For future migrations if schema changes
      }
    ),
    { 
      name: 'liked-items-store', // DevTools identifier
      /**
       * Learning: Store separation for cleaner debugging
       * Liked items are separate from search state for better organization
       */
    }
  )
) 