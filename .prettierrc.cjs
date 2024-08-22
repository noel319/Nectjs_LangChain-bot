module.exports = {
  bracketSpacing: false,
  jsxBracketSameLine: true,
  singleQuote: true,
  trailingComma: 'es5',
  semi: false,
  printWidth: 100,
  tabWidth: 2,
  useTabs: false,
  importOrder: ['^\\u0000', '^@?\\w', '^[^.]', '^\\.'],
  importOrderSeparation: true,
  plugins: ["@trivago/prettier-plugin-sort-imports", "prettier-plugin-tailwindcss"]
};