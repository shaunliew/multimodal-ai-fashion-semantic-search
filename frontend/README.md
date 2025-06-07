# Fashion Search Frontend

A simple web interface for the AI-powered fashion semantic search system. This frontend connects to the FastAPI backend to provide an easy-to-use search interface for finding fashion items using text descriptions or images.

## 🎯 What It Does

This web app provides a clean interface to:
- **Text Search**: Type descriptions like "red summer dress" to find matching items
- **Image Search**: Upload photos to find visually similar fashion items  
- **View Results**: See similarity scores and product details
- **Filter Results**: Adjust similarity thresholds and result counts

## 🔌 Backend Integration

The frontend communicates with your FastAPI backend through these main endpoints:

### API Connection
```typescript
// Connects to your FastAPI backend
const API_URL = 'http://localhost:8000'  // Your backend address

// Text search
POST /search/text
// Image search  
POST /search/image/v2
// Health check
GET /health
```

### Key Features
- **Real-time Search**: Instant results as you type or upload
- **Similarity Scores**: Visual confidence ratings (0-100%)
- **Error Handling**: Clear messages when backend is unavailable
- **Responsive Design**: Works on desktop and mobile

## 🚀 Quick Setup

### 1. Prerequisites
- **Node.js 18+** ([Download here](https://nodejs.org/))
- **Running Backend**: Make sure your FastAPI backend is running on `http://localhost:8000`

### 2. Installation
```bash
# Navigate to frontend folder
cd frontend

# Install dependencies
npm install
```

### 3. Configuration
Create a `.env.local` file:
```bash
# Your backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Run the App
```bash
# Start development server
npm run dev

# Open browser to http://localhost:3000
```

## 🖥️ Using the App

### Text Search
1. Type a description like "blue denim jacket"
2. Adjust similarity threshold (higher = more strict matching)
3. Set number of results (1-100)
4. Click "Search" to find matching items

### Image Search  
1. Click "Upload Image" or drag & drop a photo
2. Supported formats: JPG, PNG, WebP
3. Results show visually similar items with similarity scores
4. Higher scores = more similar items

### Understanding Results
- **Similarity Score**: 0-100% how similar the item is to your search
- **Product Details**: Name, brand, category, colors
- **Images**: Click to view larger versions

## 🔧 Configuration Options

### Environment Variables (`.env.local`)
```bash
# Backend API URL (required)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional: Enable debug logging
NODE_ENV=development
```

### Customizing Search Settings
The app connects to these backend endpoints with these default settings:
- **Max Results**: 20 items per search
- **Similarity Threshold**: 0.5 (50%)
- **Request Timeout**: 30 seconds
- **File Size Limit**: 10MB for image uploads

## 🛠️ Tech Stack

### Core Framework
- **Next.js 15.3** - React framework with App Router
- **React 19** - Frontend library for building user interfaces
- **TypeScript 5** - Type-safe JavaScript development

### Styling & UI
- **TailwindCSS 4** - Utility-first CSS framework
- **Radix UI** - Headless accessible UI components
  - `@radix-ui/react-dialog` - Modal dialogs
  - `@radix-ui/react-label` - Form labels
  - `@radix-ui/react-progress` - Progress bars
  - `@radix-ui/react-separator` - Visual separators
  - `@radix-ui/react-slot` - Component composition
  - `@radix-ui/react-tabs` - Tab interfaces
- **Lucide React** - Modern icon library
- **next-themes** - Dark/light theme support

### State Management & Data
- **Zustand 5** - Lightweight state management
- **TanStack Query 5** - Server state management and caching
- **TanStack Query DevTools** - Development debugging tools

### HTTP & API
- **Axios 1.9** - HTTP client for backend API calls

### Utilities & Helpers
- **clsx** - Conditional className utility
- **tailwind-merge** - TailwindCSS class merging
- **class-variance-authority** - Component variant management
- **file-saver** - File download functionality
- **sonner** - Toast notifications
- **vaul** - Drawer/sheet components

### Development Tools
- **ESLint 9** - Code linting and formatting
- **ESLint Config Next** - Next.js specific linting rules

## 📁 Project Structure
```
frontend/
├── src/
│   ├── app/                 # Main pages
│   ├── components/          # UI components  
│   │   └── image-search/    # Search interface
│   ├── lib/                 # API connection code
│   └── types/               # TypeScript definitions
├── public/                  # Static files
└── package.json             # Dependencies
```

## 🔍 Troubleshooting

### Common Issues

**"Cannot connect to API"**
- Make sure your FastAPI backend is running on `http://localhost:8000`
- Check the `NEXT_PUBLIC_API_URL` in `.env.local`
- Try accessing `http://localhost:8000/health` in your browser

**"Search takes too long"**
- This is normal for AI processing (5-30 seconds)
- Make sure your backend has enough memory (8GB+)
- Try smaller images or fewer results

**"No results found"**  
- Lower the similarity threshold (try 0.3 instead of 0.7)
- Make sure your backend database has fashion items
- Check that the backend vector search index is configured

**Images won't upload**
- Check file size (must be under 10MB)
- Use JPG, PNG, or WebP format
- Clear browser cache and try again

## 🚦 Health Check

The app automatically checks if your backend is available:
- **Green dot**: Backend connected and working
- **Red dot**: Backend unavailable (check if it's running)
- **Yellow dot**: Backend slow or having issues

## 📊 Performance Tips

### For Better Search Results
- Use clear, well-lit product photos
- Try different search terms if no results
- Adjust similarity threshold based on needs
- Use specific descriptions ("red floral dress" vs "dress")

### For Faster Performance  
- Keep image files under 5MB
- Limit results to 10-20 items
- Make sure backend has GPU acceleration enabled
- Use wired internet connection for large uploads

## 🔄 Development Commands

```bash
# Start development server
npm run dev

# Build for production  
npm run build

# Start production server
npm start

# Check for code issues
npm run lint
```

## 🤝 Need Help?

1. **Backend Issues**: Check the backend README.md for API troubleshooting
2. **Frontend Issues**: Check browser console (F12) for error messages
3. **Performance**: Monitor your backend logs for slow queries

---

**Simple Goal**: This frontend just needs to connect to your FastAPI backend and provide a clean interface for semantic fashion search. Focus on getting the backend working first, then the frontend should work automatically.
