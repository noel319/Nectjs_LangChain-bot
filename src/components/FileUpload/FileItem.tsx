import {X} from 'lucide-react'

import {cn} from '@/lib/utils'

import {Label} from '../ui/label'

type Props = {
  name: string
  onRemove: () => void
  className?: string
}
const FileItem = ({name, className, onRemove}: Props) => {
  return (
    <div className={cn('flex items-center', className)}>
      <Label className="line-clamp-1">{name}</Label>
      <X size={15} className="w-16 cursor-pointer" onClick={onRemove} />
    </div>
  )
}

export default FileItem
