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
        baseUrl: "http://cs-489-load-balancing-545464512.us-west-1.elb.amazonaws.com"
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
        detect_v2: build.mutation<detectModelResponse, detectModelRequest>({
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

export const { useDetectMutation, usePredictMutation, useLazyRootQuery } = modelsAPI;