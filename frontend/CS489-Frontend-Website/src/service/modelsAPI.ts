import type { 
  detectModelRequest, 
  detectModelResponse, 
  getRequest, 
  getResponse, 
  predictModelRequest, 
  predictModelResponse 
} from "@/types/service/modelsAPI";
import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

export const modelsAPI = createApi({
    reducerPath: "modelsAPI",
    tagTypes: [],
    baseQuery: fetchBaseQuery({
        baseUrl: "http://127.0.0.1:8000"
    }),
    endpoints: (build) => ({
        predict: build.mutation<predictModelResponse, predictModelRequest>({
            query: (request) => {
                const formData = new FormData();
                formData.append('image', request.image);
                
                return {
                    url: `/predict`,
                    method: "POST",
                    body: formData,
                };
            }
        }),
        detect: build.mutation<detectModelResponse, detectModelRequest>({
            query: (request) => {
                const formData = new FormData();
                formData.append('image', request.image);
                
                return {
                    url: `/detect`,
                    method: "POST",
                    body: formData,
                };
            }
        }),
        detectV2: build.mutation<detectModelResponse, detectModelRequest>({
            query: (request) => {
                const formData = new FormData();
                formData.append('image', request.image);
                
                return {
                    url: `/detect_v2`,
                    method: "POST",
                    body: formData,
                };
            }
        }),
        root: build.query<getResponse, getRequest>({
            query: () => ({
                url: `/`,
                method: "GET",
            })
        })
    })
});

export const { useDetectMutation, usePredictMutation, useLazyRootQuery, useDetectV2Mutation } = modelsAPI;