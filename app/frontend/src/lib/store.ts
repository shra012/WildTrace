// A tiny external store (useSyncExternalStore based) so high-frequency updates only re-render the
// components that select the changed slice.

import { useSyncExternalStore } from "react";

export interface Store<S> {
  get(): S;
  set(update: Partial<S> | ((state: S) => Partial<S>)): void;
  subscribe(listener: () => void): () => void;
  readonly initial: S;
}

export function createStore<S extends object>(initial: S): Store<S> {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    initial,
    get: () => state,
    set(update) {
      const patch = typeof update === "function" ? update(state) : update;
      let changed = false;
      for (const key of Object.keys(patch) as (keyof S)[]) {
        if (!Object.is(state[key], patch[key])) {
          changed = true;
          break;
        }
      }
      if (!changed) return;
      state = { ...state, ...patch };
      for (const listener of listeners) listener();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

/** Selectors must return stable references (primitives or objects already in the state). */
export function useStore<S extends object, T>(store: Store<S>, selector: (state: S) => T): T {
  return useSyncExternalStore(
    store.subscribe,
    () => selector(store.get()),
    () => selector(store.initial),
  );
}
