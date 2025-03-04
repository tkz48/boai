import { useState } from 'react';
import { callSidecar } from '~/routes/api.sidecar';
import { createScopedLogger } from '~/utils/logger';

const logger = createScopedLogger('useSidecar');

interface UseSidecarOptions {
  onError?: (error: Error) => void;
}

/**
 * Hook for interacting with Sidecar functionality
 */
export function useSidecar(options: UseSidecarOptions = {}) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  /**
   * Call a Sidecar endpoint
   */
  const callEndpoint = async <T = any>(endpoint: string, payload: any = {}): Promise<T | null> => {
    setIsLoading(true);
    setError(null);

    try {
      logger.debug(`Calling Sidecar endpoint: ${endpoint}`);
      const response = await callSidecar(endpoint, payload);
      return response as T;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      logger.error(`Error calling Sidecar endpoint ${endpoint}:`, error);
      setError(error);
      
      if (options.onError) {
        options.onError(error);
      }
      
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Get repository information using Sidecar
   */
  const getRepoInfo = async (repoPath: string) => {
    return callEndpoint('/repomap/info', { path: repoPath });
  };

  /**
   * Analyze code using Sidecar
   */
  const analyzeCode = async (code: string, language: string) => {
    return callEndpoint('/tree_sitter/documentation_parsing', { 
      code, 
      language 
    });
  };

  /**
   * Get code suggestions using Sidecar
   */
  const getCodeSuggestions = async (code: string, language: string, cursor: number) => {
    return callEndpoint('/inline_completion', {
      code,
      language,
      cursor
    });
  };

  /**
   * Edit file using Sidecar
   */
  const editFile = async (filePath: string, content: string) => {
    return callEndpoint('/file/edit_file', {
      path: filePath,
      content
    });
  };

  /**
   * Use Sidecar's agentic capabilities
   */
  const useAgent = async (prompt: string, files: Array<{ path: string, content: string }>) => {
    return callEndpoint('/agentic/agent_session_chat', {
      prompt,
      files
    });
  };

  return {
    isLoading,
    error,
    callEndpoint,
    getRepoInfo,
    analyzeCode,
    getCodeSuggestions,
    editFile,
    useAgent
  };
}