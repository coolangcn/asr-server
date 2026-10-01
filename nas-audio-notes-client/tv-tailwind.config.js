/** TV 放映厅页静态 Tailwind 编译配置
 *  重新生成: npx tailwindcss@3.4.17 -c tv-tailwind.config.js -i static/pb_tv.src.css -o static/pb_tv.css --minify
 *  与 CDN 版配置保持一致 (picturebook_tv.html 原 tailwind.config)
 */
module.exports = {
  content: ['./templates/picturebook_tv.html'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'sans-serif'] },
      colors: {
        ink: '#07111f', panel: '#101827', line: '#223047',
        mint: '#22c55e', sky: '#38bdf8', amber: '#f59e0b'
      }
    }
  }
};
