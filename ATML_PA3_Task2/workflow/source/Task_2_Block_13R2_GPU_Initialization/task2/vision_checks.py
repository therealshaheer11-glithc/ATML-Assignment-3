"""Check pinned SigLIP preprocessing values without depending on container type."""
import math


def channel_values(value,field):
    if isinstance(value,(str,bytes)):
        raise ValueError(field+' must contain three finite numeric channel values.')
    try:
        result=tuple(float(item) for item in value)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError(field+' must contain three finite numeric channel values.') from error
    if len(result)!=3 or not all(math.isfinite(item) for item in result):
        raise ValueError(field+' must contain three finite numeric channel values.')
    return result


def validate_siglip_processor(processor,original):
    """Validate values only. Do not rewrite processor attributes or its source JSON."""
    expected_channels=(0.5,0.5,0.5)
    for field in ('image_mean','image_std'):
        source=channel_values(original.get(field),field+' in original configuration')
        observed=channel_values(getattr(processor,field,None),field+' in instantiated processor')
        if source!=expected_channels or observed!=source:
            raise ValueError(f'SigLIP {field} differs from the pinned configuration: observed {observed!r}, source {source!r}.')
    size=dict(processor.size)
    if original['size']!={'height':224,'width':224} or size!=original['size']:
        raise ValueError('SigLIP image size differs from the pinned 224 by 224 configuration.')
    if int(original['resample'])!=3 or int(processor.resample)!=3:
        raise ValueError('SigLIP resize must retain the original bicubic resampling.')
    for field in ('do_resize','do_rescale','do_normalize'):
        if original.get(field) is not True or getattr(processor,field,None) is not True:
            raise ValueError('SigLIP '+field+' differs from its enabled original setting.')
    if original['rescale_factor']!=1/255 or float(processor.rescale_factor)!=original['rescale_factor']:
        raise ValueError('SigLIP rescale factor differs from the original 1/255 value.')
    if processor.backend!='pil':
        raise ValueError('This feature recipe requires the pinned PIL backend.')
    if getattr(processor,'do_center_crop',False) or getattr(processor,'do_pad',False):
        raise ValueError('Unexpected cropping or padding in the instantiated SigLIP processor.')
    return {'status':'passed','image_mean':list(channel_values(processor.image_mean,'image_mean')),
        'image_std':list(channel_values(processor.image_std,'image_std')),
        'mean_container_type':type(processor.image_mean).__name__,
        'std_container_type':type(processor.image_std).__name__,
        'size':size,'resample':int(processor.resample),'rescale_factor':float(processor.rescale_factor),
        'backend':processor.backend,'processor_attributes_modified':False}
