'use client'

import {create} from 'zustand'
import {createJSONStorage, persist} from 'zustand/middleware'

type Creator = 'USER' | 'AI'

type Message = {
  id: string
  text: string
  createdAt: Date
  creator: Creator
}

type MessageState = {
  messages: Message[]
  setMessages: (creator: Creator, message: string) => void
  clearMessages: () => void
}

const generateRandomId = () => Math.random().toString(36).substring(2, 9)

export const useMessages = create(
  persist<MessageState>(
    (set, get: any) => ({
      messages: get()?.messages || [],
      clearMessages: () => {
        return set({
          messages: [],
        })
      },
      setMessages: (creator: Creator, message: string) => {
        return set(() => {
          const storedMessages = get().messages

          const lastMessage = storedMessages[storedMessages.length - 1]
          // if (!lastMessage) return storedMessages

          if (creator === 'AI' && lastMessage.creator === 'AI') {
            return {
              messages: [...storedMessages.slice(0, -1), {...lastMessage, text: message}],
            }
          }

          return {
            messages: [
              ...storedMessages,
              {id: generateRandomId(), text: message, createdAt: new Date(), creator},
            ],
          }
        })
      },
    }),
    {
      name: 'messages', // name of the item in the storage (must be unique)
      storage: createJSONStorage(() => sessionStorage), // (optional) by default, 'localStorage' is used
      skipHydration: true,
    }
  )
)

type FileState = {
  file: File | null
  setFile: (file: File) => void
  clear: () => void
}

export const useFile = create(
  persist<FileState>(
    (set, get: any) => ({
      file: get()?.file || null,
      clear: () => {
        return set({
          file: null,
        })
      },
      setFile: (file: FileState['file']) => {
        return set(() => {
          return {
            file,
          }
        })
      },
    }),
    {
      name: 'file',
      storage: createJSONStorage(() => sessionStorage),
      skipHydration: true,
    }
  )
)
