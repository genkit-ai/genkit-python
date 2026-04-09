[Genkit](https://genkit.dev) is an open-source framework for building full-stack AI-powered applications, built and used in production by Google's Firebase. It provides SDKs for multiple programming languages with varying levels of stability:

- **JavaScript/TypeScript**: Production-ready with full feature support
- **Go**: Production-ready with full feature support
- **Python (Alpha)**: Early development with core functionality

It offers a unified interface for integrating AI models from providers like [Google](https://genkit.dev/docs/plugins/google-genai), [OpenAI](https://genkit.dev/docs/plugins/openai), [Anthropic](https://thefireco.github.io/genkit-plugins/docs/plugins/genkitx-anthropic), [Ollama](https://genkit.dev/docs/plugins/ollama/), and more. Rapidly build and deploy production-ready chatbots, automations, and recommendation systems using streamlined APIs for multimodal content, structured outputs, tool calling, and agentic workflows.

Get started with just a few lines of code:

```ts
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({ plugins: [googleAI()] });

const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: 'Why is Firebase awesome?'
});
```

## Explore & build with Genkit

Play with AI sample apps, with visualizations of the Genkit code that powers
them, at no cost to you.

[Explore Genkit by Example](https://examples.genkit.dev)