export class ApiError extends Error {
  readonly status: number;
  /** True when the backend could not be reached at all (network error or timeout). */
  readonly offline: boolean;

  constructor(message: string, status = 0, offline = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.offline = offline;
  }
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}
