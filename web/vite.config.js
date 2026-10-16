import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` proxies the Flask API (python server.py --port 5000).
// Prod: `npm run build` -> web/dist, served by server.py at "/".
const backend = process.env.MORPHIX_BACKEND || "http://127.0.0.1:5000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": backend, "/runs": backend },
  },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
