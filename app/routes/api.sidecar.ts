import { type ActionFunctionArgs, json } from '@remix-run/cloudflare';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('api.sidecar');
const SIDECAR_PORT = 3000; // Default port for Sidecar service

/**
 * Handles requests to the Sidecar service
 */
export async function action({ request }: ActionFunctionArgs) {
  try {
    const { endpoint, payload } = await request.json();
    
    if (!endpoint) {
      return json({ error: 'Missing endpoint parameter' }, { status: 400 });
    }
    
    logger.info(`Forwarding request to Sidecar endpoint: ${endpoint}`);
    
    const sidecarUrl = `http://localhost:${SIDECAR_PORT}/api${endpoint}`;
    
    const response = await fetch(sidecarUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload || {}),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      logger.error(`Sidecar request failed: ${response.status} - ${errorText}`);
      return json({ error: `Sidecar request failed: ${response.status}` }, { status: response.status });
    }
    
    // Check if response is a stream
    const contentType = response.headers.get('Content-Type');
    if (contentType && contentType.includes('text/event-stream')) {
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    }
    
    const data = await response.json();
    return json(data);
  } catch (error) {
    logger.error('Error in Sidecar API:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}

/**
 * Helper function to make requests to the Sidecar service
 */
export async function callSidecar(endpoint: string, payload: any = {}) {
  try {
    const response = await fetch('/api/sidecar', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ endpoint, payload }),
    });
    
    if (!response.ok) {
      throw new Error(`Sidecar request failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error calling Sidecar:', error);
    throw error;
  }
}