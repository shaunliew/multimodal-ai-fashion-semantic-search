/**
 * @fileoverview Modal Store - Zustand state management for modals and overlays
 * @learning Demonstrates centralized UI state management for better user experience
 * @concepts Modal state management, TypeScript generics, reusable patterns
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { SearchResult } from './imageSearchStore'
import type { FashionProduct } from '@/types/api'

/**
 * Learning: Generic modal state interface for reusability
 * This pattern allows the same store to manage different types of modals
 * while maintaining type safety with TypeScript generics
 */
interface ModalState<T = unknown> {
  // Modal visibility state
  isOpen: boolean
  modalType: string | null
  
  // Generic data for modal content
  data: T | null
  
  // Actions for modal management
  openModal: (type: string, data?: T) => void
  closeModal: () => void
  updateModalData: (data: Partial<T>) => void
}

/**
 * Learning: Specific interface for result details modal
 * This demonstrates how to extend generic patterns for specific use cases
 */
interface ResultDetailsModalState {
  isResultDetailsOpen: boolean
  selectedResult: SearchResult | null
  currentTab: 'analysis' | 'features' | 'vectors'
  
  // Specific actions for result details modal
  openResultDetails: (result: SearchResult) => void
  closeResultDetails: () => void
  setActiveTab: (tab: 'analysis' | 'features' | 'vectors') => void
}

/**
 * Learning: Specific interface for product details modal (Fashion Product from API)
 * This demonstrates how to handle different data types in the same modal system
 */
interface ProductDetailsModalState {
  isProductDetailsOpen: boolean
  selectedProduct: FashionProduct | null
  
  // Specific actions for product details modal
  openProductDetails: (product: FashionProduct) => void
  closeProductDetails: () => void
}

/**
 * Learning: Combined modal store interface
 * This shows how to compose multiple modal patterns in one store
 */
interface CombinedModalState extends ModalState, ResultDetailsModalState, ProductDetailsModalState {}

/**
 * Learning: Zustand store for centralized modal management
 * Benefits of centralized modal state:
 * - Consistent modal behavior across the app
 * - Easier to manage modal stacking and overlays
 * - Better debugging with DevTools
 * - Prevents prop drilling for modal state
 */
export const useModalStore = create<CombinedModalState>()(
  devtools(
    (set, get) => ({
      // Generic modal state
      isOpen: false,
      modalType: null,
      data: null,

      // Result details modal specific state
      isResultDetailsOpen: false,
      selectedResult: null,
      currentTab: 'analysis',

      // Product details modal specific state
      isProductDetailsOpen: false,
      selectedProduct: null,

      // Generic modal actions
      openModal: (type, data) => {
        /**
         * Learning: Generic modal opening with type safety
         * This pattern allows multiple modal types while maintaining state consistency
         */
        set({ 
          isOpen: true, 
          modalType: type, 
          data 
        })
      },

      closeModal: () => {
        /**
         * Learning: Clean modal closure
         * Always reset all modal state to prevent memory leaks and state pollution
         */
        set({ 
          isOpen: false, 
          modalType: null, 
          data: null 
        })
      },

      updateModalData: (newData) => {
        const currentData = get().data
        set({ 
          data: currentData ? { ...currentData, ...newData } : newData 
        })
      },

      // Result details modal specific actions
      openResultDetails: (result) => {
        /**
         * Learning: Specific modal opening with domain logic
         * This demonstrates how to create focused actions for specific modal types
         * while maintaining the generic modal pattern
         */
        set({ 
          isResultDetailsOpen: true,
          selectedResult: result,
          currentTab: 'analysis', // Always start with analysis tab
          
          // Also set generic modal state for consistency
          isOpen: true,
          modalType: 'result-details',
          data: result
        })
      },

      closeResultDetails: () => {
        /**
         * Learning: Specific modal closure with cleanup
         * This ensures both specific and generic modal state are properly reset
         */
        set({ 
          isResultDetailsOpen: false,
          selectedResult: null,
          currentTab: 'analysis',
          
          // Also clear generic modal state
          isOpen: false,
          modalType: null,
          data: null
        })
      },

      setActiveTab: (tab) => {
        /**
         * Learning: Tab state management within modal
         * This demonstrates how to manage sub-states within a modal context
         * Useful for complex modals with multiple views
         */
        set({ currentTab: tab })
      },

      // Product details modal specific actions
      openProductDetails: (product) => {
        /**
         * Learning: Product details modal for API Fashion Products
         * This demonstrates handling different data types in the modal system
         */
        set({ 
          isProductDetailsOpen: true,
          selectedProduct: product,
          
          // Also set generic modal state for consistency
          isOpen: true,
          modalType: 'product-details',
          data: product
        })
      },

      closeProductDetails: () => {
        /**
         * Learning: Product details modal closure with cleanup
         * Ensures both specific and generic modal state are properly reset
         */
        set({ 
          isProductDetailsOpen: false,
          selectedProduct: null,
          
          // Also clear generic modal state
          isOpen: false,
          modalType: null,
          data: null
        })
      }
    }),
    { 
      name: 'modal-store', // DevTools identifier
      /**
       * Learning: Store partitioning for better debugging
       * This helps separate modal state from other application state in DevTools
       */
    }
  )
)

/**
 * Learning: Convenience hooks for specific modal types
 * These provide a cleaner API for components while still using the centralized store
 */

/**
 * Hook specifically for result details modal
 * This demonstrates how to create focused APIs from a centralized store
 */
export const useResultDetailsModal = () => {
  const {
    isResultDetailsOpen,
    selectedResult,
    currentTab,
    openResultDetails,
    closeResultDetails,
    setActiveTab
  } = useModalStore()

  return {
    isOpen: isResultDetailsOpen,
    result: selectedResult,
    activeTab: currentTab,
    openModal: openResultDetails,
    closeModal: closeResultDetails,
    setActiveTab
  }
}

/**
 * Hook specifically for product details modal (Fashion Product from API)
 * This demonstrates how to create focused APIs from a centralized store
 */
export const useProductDetailsModal = () => {
  const {
    isProductDetailsOpen,
    selectedProduct,
    openProductDetails,
    closeProductDetails
  } = useModalStore()

  return {
    isOpen: isProductDetailsOpen,
    product: selectedProduct,
    openModal: openProductDetails,
    closeModal: closeProductDetails
  }
}

/**
 * Learning: Generic modal hook
 * This shows how to provide access to the generic modal functionality
 */
export const useGenericModal = () => {
  const {
    isOpen,
    modalType,
    data,
    openModal,
    closeModal,
    updateModalData
  } = useModalStore()

  return {
    isOpen,
    modalType,
    data,
    openModal,
    closeModal,
    updateModalData
  }
} 