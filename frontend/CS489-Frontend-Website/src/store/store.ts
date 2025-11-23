import { configureStore } from "@reduxjs/toolkit";
import { modelsAPI } from "@/service/modelsAPI";
export const store = configureStore({
    reducer: {
        [modelsAPI.reducerPath]: modelsAPI.reducer
    },  
    middleware: (getDefaultMiddleware) => 
        getDefaultMiddleware()
            .concat(modelsAPI.middleware)
})

export type RootState = ReturnType<typeof store.getState>;

export type AppDispatch = typeof store.dispatch;