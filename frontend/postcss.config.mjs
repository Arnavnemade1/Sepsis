/**
 * Tailwind v4 only. NOT `tailwindcss: {}` (that is the v3 plugin name), and no
 * autoprefixer — v4 handles vendor prefixing through Lightning CSS.
 */
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
