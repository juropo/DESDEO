import storybook from 'eslint-plugin-storybook';
import prettier from 'eslint-config-prettier';
import js from '@eslint/js';
import { includeIgnoreFile } from '@eslint/compat';
import svelte from 'eslint-plugin-svelte';
import ts from 'typescript-eslint';
import globals from 'globals';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import svelteConfig from './svelte.config.js';
import { defineConfig } from 'eslint/config';
import ts from 'typescript-eslint';

const gitignorePath = fileURLToPath(new URL('./.gitignore', import.meta.url));
const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(
	includeIgnoreFile(gitignorePath),

	js.configs.recommended,

	// TypeScript with type-aware rules
	...ts.configs.recommendedTypeChecked,
	...ts.configs.stylisticTypeChecked,

	// Svelte (FLAT!) + Prettier compat
	...svelte.configs['flat/recommended'],
	prettier,
	...svelte.configs.prettier,

	// Default: browser globals for client code
	{
		languageOptions: {
			globals: globals.browser,
			parserOptions: {
				project: ['./tsconfig.json'],
				tsconfigRootDir: __dirname,
				ecmaVersion: 'latest',
				sourceType: 'module'
			}
		},
		rules: {
			// avoid false positives when using TS
			'no-undef': 'off'
		}
	},

  // Files outside tsconfig.json: lint without type information
  {
    files: ['*.{js,mjs,ts}', '.storybook/**', 'e2e/**', 'tests/**'],
    ...ts.configs.disableTypeChecked
  },

  // Make sure Svelte blocks use TS parser
  {
    files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
    languageOptions: {
      parserOptions: {
        parser: ts.parser,
        extraFileExtensions: ['.svelte'],
        svelteConfig
      }
    }
  },

	// Node env for server-only files
	{
		files: [
			'**/+layout.server.{js,ts}',
			'**/+page.server.{js,ts}',
			'**/+server.{js,ts}',
			'src/hooks.server.{js,ts}',
			'src/routes/**/*.server.{js,ts}'
		],
		languageOptions: {
			globals: globals.node
		},
		rules: {
			'no-console': 'off'
		}
	},

	// Root-level tooling config (eslint.config.js itself, vite.config.ts, etc.) isn't part of
	// tsconfig.json's `include` (only `src/**`), so it has no type information to type-aware-lint
	// against — drop down to syntax-only rules for just these files.
	{
		files: ['*.config.{js,ts,mjs}'],
		...ts.configs.disableTypeChecked,
		languageOptions: {
			...ts.configs.disableTypeChecked.languageOptions,
			globals: globals.node
		}
	},

	// Storybook
	storybook.configs['flat/recommended']
);
