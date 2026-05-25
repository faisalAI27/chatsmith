import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
const missingSupabaseEnv = !supabaseUrl || !supabaseKey;

export const supabaseConfigError = missingSupabaseEnv
  ? "Missing Supabase config. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in frontend/.env."
  : "";

export const supabase = missingSupabaseEnv ? null : createClient(supabaseUrl, supabaseKey);
